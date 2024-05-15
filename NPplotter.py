import numpy as np
from plotter import plot_psth, plot_psth_with_rasters
import shutil
from postPhyAnal import *
import time
from matplotlib import pyplot as plt
from matplotlib import gridspec
from npyx import *
import multiprocessing as mp
import pandas as pd



from npyx.spk_wvf import get_peak_chan, wvf, templates
from npyx.plot import plot_wvf, get_peak_chan
# matplotlib.use('Qt5Agg')

def getEventData(pathtoses, SesName, g=0, evIDs=(1,2,3,4)):
    evDic = {}
    evpath = r'{}\{}_g{}'.format(pathtoses, SesName, g)
    # event_ts = np.load(os.path.join(evpath,'timestamps.npy'), mmap_mode='r')
    # eventsID = np.load(os.path.join(evpath, 'channel_states.npy'), mmap_mode='r')
    eventsID_tastes = np.array([])
    synchronized_event_ts = np.array([])

# IDs of events indicates the taste?
    for i in evIDs:
        evfile = r'{}\{}_g{}_tcat.nidq.XD_0_{}_0_corr.txt'.format(evpath, SesName, g, i)
        arr = np.loadtxt(evfile, 'float')
        synchronized_event_ts = np.append(synchronized_event_ts, arr.tolist())
        eventsID_tastes = np.append(eventsID_tastes, [i]*len(arr))

    data = {'ID': eventsID_tastes.flatten(), 'sync_timestamps': synchronized_event_ts.flatten() }
    ev = pd.DataFrame(data=data)
    return ev

def make_dic2(tastes_dic, br1, br2):
# br1 and br2 are the borders of the epoch from which trials are gatherd
    tr_taste_dic = {}
    for key in tastes_dic.keys():
        trials = np.asarray(tastes_dic[key])
        this_trial = trials[(trials>=br1) & (trials<=br2)]
        tr_taste_dic[key] = this_trial
    return tr_taste_dic

def findSessBorders(inp):
    a = [item for sublist in inp for item in sublist] # shove all times in 1 list
    a = np.sort(a)  #sort the times
    df = np.diff(a)
    inds = np.where(df > 1000)[0]  #indices of the 1st event that is ~16min apart from previous
    if len(inds) == 0:
        return np.array([0, a[-1]+5])
    else:
        b = [a[i] for i in inds]  #events times from a that fullfil the time-gap criteria
        # add 0 to first position and a time at last position that is 45min after(why?)
        b.insert(0, int(0))
        b.append(a[-1]+2700)

        return np.asarray(b)

def findSessBorders_byday(inp, daychangetimes):
    taste_events_sec = [item for sublist in inp for item in sublist]  # shove all times in 1 list
    taste_events_sec = np.sort(taste_events_sec)  # sort the times

    days_sesions_boards = []
    take1st = taste_events_sec[0]
    daychangetimes = np.append(daychangetimes, daychangetimes[-1]+2*3600)  #add a time of 2hrs after the beginning of last day
    # print(daychangetimes)
    for daystart in daychangetimes:
        # daystartint = int(daystart)
        print('Got indices for a new day, range is between {}-{}'.format(take1st, daystart))
        dayevents_inx = np.where(np.logical_and(taste_events_sec >= take1st , taste_events_sec < daystart))[0]
        dayevents_times = taste_events_sec[dayevents_inx]
        eventsboards_inx = np.where(np.diff(dayevents_times) > 1000)[0] # > 16 min
        eventsboards_times = []
        if (eventsboards_inx.size==0) :  #in case of 1 session/day
            eventsboards_times.append((dayevents_times[0],dayevents_times[-1]))
            eventsboards_times.append([])
        else:
            for i in range(len(eventsboards_inx)):  #indices are the last event of each session
                if i==0:
                    endofsession_inx = eventsboards_inx[i]
                    eventsboards_times.append((dayevents_times[0],dayevents_times[endofsession_inx]))
                else:
                    endofprevsess = eventsboards_inx[i-1]
                    endofsession_inx = eventsboards_inx[i]
                    eventsboards_times.append((dayevents_times[endofprevsess+1], dayevents_times[endofsession_inx]))
            endoflast = eventsboards_inx[-1]  # add the end of last session
            eventsboards_times.append((dayevents_times[endoflast+1], dayevents_times[-1]))
        days_sesions_boards.append(eventsboards_times)
        take1st = daystart

    return days_sesions_boards

def session_time_from_cta(firstevent, ctatime):
    return firstevent/3600 - ctatime

def plot_3wvf(dp, uu, color='dodgerblue', scalebar_w=5, figw=None, axlst=None):
    if (figw==None):
        figw = plt.figure()
    if (axlst==None):
        axlst = [figw.add_subplot(211), figw.add_subplot(212), figw.add_subplot(213)]

    fs = read_metadata(dp)['highpass']['sampling_rate']
    waveforms = wvf(dp, u=uu, n_waveforms=100)
    n_samples = waveforms.shape[-2]

    # center around peak channel
    peakchannel = get_peak_chan(dp, uu)
    chanStart, chanEnd = int(peakchannel - 1), int(peakchannel + 1)
    data = waveforms[:, :, chanStart:chanEnd + 1]
    data = data[~np.isnan(data[:, 0, 0]), :, :]  # filter out nan waveforms
    datam = np.mean(data, 0)
    datastd = np.std(data, 0)
    datamt = datam.T
    waveforms_std_t = datastd.T
    datamin, datamax = np.nanmin(datamt - waveforms_std_t) - 50, np.nanmax(datamt + waveforms_std_t) + 50
    t = np.linspace(0, n_samples / (fs / 1000), n_samples)

    # figw, ax = plt.subplots(3, 1, figsize=(4, 4))
    for i in range(3):
        axlst[i].plot(t, datamt[i, :])
        axlst[i].plot(t, datamt[i, :] + waveforms_std_t[i, :], linewidth=1, color=color, alpha=0.5)
        axlst[i].plot(t, datamt[i, :] - waveforms_std_t[i, :], linewidth=1, color=color, alpha=0.5)
        axlst[i].fill_between(t, datamt[i, :] - waveforms_std_t[i, :], datamt[i, :] + waveforms_std_t[i, :], color=color,
                           interpolate=True, alpha=0.2)

        axlst[i].set_ylim([datamin, datamax])
        axlst[i].set_xlim([t[0], t[-1]])
        axlst[i].axis('off')
        # add scalebar
        ylimdiff = datamax - datamin
        y_scale = int(ylimdiff * 0.3 - (ylimdiff * 0.3) % 10)
        axlst[2].plot([0, 1], [datamin, datamin], c='k', lw=scalebar_w)
        axlst[2].text(0.5, datamin - 0.05 * ylimdiff, '1 ms', size=6, va='top', ha='center')
        axlst[2].plot([0, 0], [datamin, datamin + y_scale], c='k', lw=scalebar_w)
        axlst[2].text(-0.1, datamin + y_scale * 0.5, f'{y_scale}\u03bcV', size=6,va='center', ha='right')
    axlst[0].set_title('Waveforms in peak channel\nand around',fontsize=8)
    return figw

def plotUnitPSTHbyDay(data):
    sessListBorders = data[0]
    pathtopick = data[1]
    uu = data[2]
    dep = data[3]
    tastes_dic = data[4]
    figs_folder = data[5]
    tlist = data[6]
    ctatime = data[7]
    expdays = data[8]
    dppath = data[9]

    numrows = len(sessListBorders)
    numcols = len(max(sessListBorders, key=len))
    fig = plt.figure(figsize=(15, 15))
    gses = gridspec.GridSpec(numrows, numcols, figure=fig, wspace=0.3, hspace=0.3)

    lims = []
    axdic = {}
    # I shoudl change this to Shared memory
    sptimes = np.load(pathtopick + r'\spike_seconds.npy', mmap_mode='r')
    clust = np.load(pathtopick + r'\spike_clusters.npy', mmap_mode='r')
    st = sptimes[clust == uu].flatten()  # get the spike times for a specific cluster
    for r in range(numrows):
        currentday = sessListBorders[r]
        for col in range(len(currentday)):
            print('Now working grid in row {} in coumn {}'.format(r, col))
            if (col == 1) and (r == 3):
                wnew_gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gses[r, col])
                wax1, wax2, wax3 = fig.add_subplot(wnew_gs[0]), fig.add_subplot(wnew_gs[1]), fig.add_subplot(wnew_gs[2])
                plot_3wvf(dppath, uu, figw=fig, axlst=[wax1, wax2, wax3])

            else:
                new_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gses[r, col], hspace=0.02)
                ax1 = fig.add_subplot(new_gs[0])
                ax2 = fig.add_subplot(new_gs[1])

                tr_taste_dic = make_dic2(tastes_dic, currentday[col][0], currentday[col][1])
                plot_psth_with_rasters(1, uu, spike_train=st, event_dic=tr_taste_dic, taste_list=tlist, start_time=-2,
                                       end_time=4, fig=fig, ax1=ax1, ax2=ax2, legend=False)

                axname = '{}_{}_{}'.format(expdays[r], r, col)
                sestime = session_time_from_cta(currentday[col][0], ctatime)
                axdic[axname] = [ax2, ax2.get_ylim(), sestime]

            if col == 0:
                ax1.set_xlabel('')
                ax1.set_ylabel('Trails', fontsize=6)
                ax1.set_yticklabels([])
                ax1.set_xticks([])
                ax1.set_xlim(-2.5, 4.5)
                ax2.set_xlabel('Time (s)', fontsize=6)
                ax2.set_ylabel('FR (sp/s)', fontsize=6)
                # ax2.set_xticklabels([-2, 0, 2, 4])
                ax2.set_xlim(-2.5, 4.5)
            else:
                ax1.set_xlabel('')
                ax1.set_ylabel('')
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_xlim(-2.5, 4.5)
                ax2.set_xlabel('')
                ax2.set_ylabel('')
                # ax2.set_xticklabels([-2, 0, 2, 4])
                ax2.set_xlim(-2.5, 4.5)

    lims = [x[1] for x in axdic.values()]
    nplims = np.reshape(lims, (len(lims), 2))
    maxlim = nplims[:, 1].max()
    minlim = nplims[:, 0].min()
    for ax in axdic.keys():
        curax = axdic[ax]
        curax[0].set_ylim(minlim, maxlim)
        ypos = maxlim - maxlim / 4
        curax[0].text(2, ypos, 't={:.0f}'.format(curax[2]), fontsize='xx-small')
        curax[0].tick_params(axis='both', which='major', labelsize=6)
    pos = 0.875
    for day in expdays:
        plt.text(0, pos, day, transform=fig.transFigure, fontsize=8)
        pos -= 0.25

    plt.rcParams.update({'font.size': 6})
    fig.suptitle('Cluster {}, Depth {}mm from tip'.format(str(uu), str(dep)), fontsize=18)
    fname = '{}/depth_{}_clust_{}'.format(figs_folder, str(dep), str(uu))
    print('Saving to file: {}'.format(fname))
    plt.savefig(fname + '.svg', format='svg', dpi=300)
    plt.savefig(fname + '.png', format='png', dpi=300)
    plt.close(fig)
    return 1

def plotNPPSTH(path_to_ses,SesName, ctatime, expdays, g=0, imec=0, KKsorter='kilosort3', evIDs=(1,2,3,4), breaktosess=True):

    # gu = getGoodUnits(dpks)
    # gudata = getUnitsData(dpks, gu)
    # meta = read_metadata(dp)
    # aa = npyx.get_waveforms(dpks, 662, 10)
    # ses_num = 6  # number of taste sessions
    # trials_per_ses = 15 # number of taste trials per each session
    # lfp_samples, lfp_times, lfp_cont_sync_ts = getLFP(path_to_ses, RecNode, exp, recording, imec)
    dp = combineToGLXData(path_to_ses, SesName, g, imec)
    dpks = combineToGLX_KS_path(path_to_ses, SesName, g, imec, KKsorter)
    ev = getEventData(path_to_ses, SesName, g, evIDs=evIDs)

    t1s = ev[ev['ID'] == 1]['sync_timestamps'].tolist()
    t2s = ev[ev['ID'] == 2]['sync_timestamps'].tolist()
    t3s = ev[ev['ID'] == 3]['sync_timestamps'].tolist()
    t4s = ev[ev['ID'] == 4]['sync_timestamps'].tolist()
    t5s = ev[ev['ID'] == 5]['sync_timestamps'].tolist()

    pathtopick = os.path.join(path_to_ses, r'{}_g{}\{}_g{}_imec{}\{}'.format(SesName, g, SesName, g, imec, KKsorter))

    df_clusgroup = pd.read_csv(pathtopick + r'\cluster_group.tsv', sep='\t', header=0)
    df_clusinfo = pd.read_csv(pathtopick + r'\cluster_info.tsv', sep='\t', header=0)
    sptimes = np.load(pathtopick + r'\spike_seconds.npy', mmap_mode='r')
    clust = np.load(pathtopick + r'\spike_clusters.npy', mmap_mode='r')
    g_unitss_gr = df_clusgroup[df_clusgroup['group'] == 'good'][['cluster_id']]
    g_unitss = df_clusinfo[df_clusinfo['cluster_id'].isin(g_unitss_gr['cluster_id'])][['cluster_id', 'depth']]

    daytimes = np.loadtxt(os.path.join(path_to_ses, r'{}_g{}\{}'.format(SesName, g, 'daytimes_uncorr.txt')))
    sessListBorders = findSessBorders_byday([t1s, t2s, t3s, t4s], daytimes)
    tastes_dic = {'water': t1s, 'sugar': t2s, 'nacl': t3s, 'CA': t4s}
    tlist = ['water', 'sugar', 'nacl', 'CA']

    figs_folder = r'{}\figss_slaves_psthonly'.format(dpks)
    if os.path.isdir(figs_folder):
        ret = input("Do you really want to delete the fig folder? press y or n")
        if ret=='y':
            shutil.rmtree(figs_folder, ignore_errors=False)
            os.mkdir(figs_folder)
    else:
        os.mkdir(figs_folder)

    print('Found {} good units, starts plotting'.format(len(g_unitss_gr)))
    # whether to break the fig
    if (breaktosess == False):
        fig = plt.figure(figsize=(15, 15))
        for uu, dep in g_unitss.values:
            st = sptimes[clust == uu].flatten()

            plot_psth_with_rasters(1, uu, st, tastes_dic, taste_list=tlist, start_time=-4,
                                   end_time=4, fig=fig)
            plt.savefig('{}/depth_{}_clust_{}.png'.format(figs_folder, str(dep), str(uu)))
            plt.clf()

    else:
        data = []
        for uu, dep in g_unitss.values:
            data.append([sessListBorders, pathtopick, uu, dep, tastes_dic, figs_folder, tlist, ctatime, expdays, dp])

        with mp.Pool(processes=38) as pool:
            results = pool.map_async(plotUnitPSTHbyDay, tuple(data), chunksize=1)
            # Make cross_correlogram
            prev_count = results._number_left
            results.wait(timeout=10)
            while not results.ready():
                t = time.localtime()
                numleft = results._number_left
                current_time = time.strftime("%H:%M:%S", t)
                print("{}:   num left: {} rate = {} res/sec".format(current_time, numleft, (prev_count - numleft) / 10))
                prev_count = numleft
                results.wait(timeout=10)
        print('going to sleep...')
        time.sleep(5)
        J_f_list = results.get()

def packNPData(pathtoses, SesName, pathtosp,
               taste_names=('water','sucrose','nacl', 'CA','quinine'),
               taste_ids=range(1,6),
               start_time=-1, end_time=4, bin_width=0.2, normalize="Hz"):

    ev = getEventData(pathtoses, SesName)
    gu = npyx.get_units(pathtosp, "good")
    assert end_time > start_time, 'start time cannot be bigger or equal to end time'
    assert (end_time - start_time) / bin_width == int((end_time - start_time) / bin_width)

    bin_amount = int((end_time - start_time) / bin_width)

    taste_events = [ev[ev['ID'] == i]['sync_timestamps'].tolist() for i in taste_ids]
    df_clusgroup = pd.read_csv(pathtosp + r'\cluster_group.tsv', sep='\t', header=0)
    df_clusinfo = pd.read_csv(pathtosp + r'\cluster_info.tsv', sep='\t', header=0)
    all_spikes = np.load(pathtosp + r'\spike_seconds.npy', mmap_mode='r')
    clust = np.load(pathtosp + r'\spike_clusters.npy', mmap_mode='r')
    sessListBorders = findSessBorders(taste_events)
    event_dic = dict(zip(taste_names, taste_events))
    ret_list = [{} for _ in range(len(sessListBorders) - 1)]
    # Do for each experimental trial (a single session with usually different tastes, several secods appart)
    for tri in range(len(sessListBorders) - 1):
        ret_list[tri]['data'] = {}
        for taste in taste_names:
            print('collecting spike times for event of taste {}'.format(taste))
            spikes_all_trials = np.array([])
            ret_list[tri]['data'][taste] = {}
            for u in gu:
                spike_train = all_spikes[clust==u]
                spike_train = spike_train[(spike_train>sessListBorders[tri]) & (spike_train<=sessListBorders[tri+1])]
                events = [event for event in event_dic[taste] if  sessListBorders[tri] <= event < sessListBorders[tri+1]]
                lengths = []
                for event in events:
                    spikes = [spike_train[i] - event for i in range(len(spike_train))  if
                          start_time < spike_train[i] - event < end_time]
                    hist1, _ = np.histogram(spikes, bin_amount, (start_time, end_time))
                    if normalize == 'Hz':
                        hist1 = hist1 / bin_width
                    # spikes_all_trials = spikes_all_trials.append(hist1)
                    lengths.append(len(hist1))
                    spikes_all_trials = np.concatenate((spikes_all_trials, hist1))
            ret_list[tri]['data'][taste]['FR'] = np.reshape(spikes_all_trials, (-1, len(gu)), 'F')
            ret_list[tri]['data'][taste]['lengths'] = lengths
    return ret_list


if __name__ == "__main__":
    path_to_ses = r'C:\Users\AnanM\OneDrive - Tel-Aviv University\Documents\TAU\data\NP\ND7_post'
    SesName = 'ND7_post'
    g = 0
    imec = 0
    KKsorter = 'kilosort3'
    pathtosp = r'C:\Users\AnanM\OneDrive - Tel-Aviv University\Documents\TAU\data\NP\ND7_post\ND7_post_g0\kilosort3'
    start_time = -1
    end_time = 5
    bin_width = 0.2

    packed_data = packNPData(path_to_ses, SesName, pathtosp,taste_names=('water','sucrose','nacl', 'CA'),
               taste_ids=range(1,5),)
    expdays = ['Hab3', 'CTA', 'Test', 'Ext']
    licl_time = 23.8  #in hrs, we'll call this t=0 of CTA
    evIDs = (1,2,3,4)
    breaktosess = True
    plotNPPSTH(path_to_ses,SesName, licl_time, expdays, g, imec, KKsorter, evIDs, breaktosess)
