"""
Created Nov 20 by Chabrun F
"""

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
import matplotlib.font_manager as font_manager
import seaborn as sns

# graphical parameters, for article figures:
pp_size = 6
legend_font_params = font_manager.FontProperties(family='Calibri',
                                                 weight='normal',
                                                 style='normal',
                                                 size=24)
paneltitle_font_params = font_manager.FontProperties(family='Calibri',
                                                     weight='bold',
                                                     style='normal',
                                                     size=24)
labels_font_params = {"fontname": "Calibri", "fontweight": "normal", "fontsize": 24}
ticks_font_params = {'family': 'Calibri', 'weight': 'normal', 'size': 22}
scatter_plot_size = 120
boxplot_mean_marker_size = 10
boxplot_lw = 4.0
boxplot_precise_lw = 1.0
boxplot_props = {
    # 'boxprops': {'edgecolor': 'black'},
    # 'medianprops': {'color': 'black'},
    # 'whiskerprops': {'color': 'black'},
    # 'capprops': {'color': 'black'},
    'flierprops': {"markersize": boxplot_mean_marker_size},
    'meanprops': {"marker": "X", "markeredgecolor": "#333333", "markerfacecolor": "#333333", "markersize": boxplot_mean_marker_size}}

rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'Calibri:italic'
rcParams['mathtext.rm'] = 'Calibri'
rcParams['mathtext.bf'] = 'Calibri:bold'
rc_params = {'font.size': 16, }
plt.rcParams.update(rc_params)


# quick function to do regression
def plot_lm(x, y, group_name, xlabel, ylabel, xticks=None, yticks=None, show_grid: bool = True,
            correlation_method=("spearman", "pearson")[1],
            n_xticks: int = 7, n_yticks: int = 7, suptitle: str = None,
            scatter_plot_custom_size: float = scatter_plot_size, scatter_kws_custom_alpha: float = 0.5,
            text_show: bool = True, text_show_sample_size: bool = True, text_bbox_alpha: float = .0, text_pos: str = "top left",
            verticalalignment=None, horizontalalignment=None,
            common_scale: bool = False, no_lm: bool = False,
            console_print_pr: bool = False,
            confidence_level: float = None):
    from scipy import stats
    from sklearn.linear_model import LinearRegression

    x = x.to_numpy() if type(x) is pd.Series else x
    x = np.array(x) if type(x) in (list, tuple) else x
    y = y.to_numpy() if type(y) is pd.Series else y
    y = np.array(y) if type(y) in (list, tuple) else y
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    xymin, xymax = min(x.min(), y.min()), max(x.max(), y.max())
    xyrange = xymax - xymin
    xrange, yrange = xmax - xmin, ymax - ymin
    if correlation_method == "pearson":  # Todo
        # r, p = scipy.stats.pearsonr(x=x, y=y)
        cor_method = stats.pearsonr
    elif correlation_method == "spearman":  # Todo
        # r, p = scipy.stats.spearmanr(a=x, b=y)
        cor_method = stats.spearmanr
    else:
        assert False, f"Unsupported {correlation_method=}"
    r, p = cor_method(x, y)
    if confidence_level is not None:
        assert (confidence_level > .5) and (confidence_level < 1), f"Unexpected {confidence_level=}"
        bts_results = stats.bootstrap((x, y), cor_method, paired=True, n_resamples=9999, confidence_level=confidence_level, random_state=1)
        r_lo = bts_results.confidence_interval[0][0]
        r_hi = bts_results.confidence_interval[1][0]
    else:
        r_lo, r_hi = None, None

    lm = LinearRegression()
    lm.fit(X=x.reshape(-1, 1), y=y.reshape(-1, 1))
    # slope, intercept = lm.coef_[0][0], lm.intercept_[0]
    if common_scale:
        xmin, xmax, xrange, ymin, ymax, yrange = xymin, xymax, xyrange, xymin, xymax, xyrange
        ax = sns.lineplot(x=[0, xymax], y=[0, xymax], linestyle='--', color="#aaaaaa")
        scatter_kws = {"color": "black", "alpha": scatter_kws_custom_alpha, "s": scatter_plot_custom_size}
        if no_lm:
            sns.scatterplot(x=x, y=y, **scatter_kws, ax=ax)
        else:
            sns.regplot(x=x, y=y, scatter_kws=scatter_kws, line_kws={"color": "red"}, ax=ax)
    else:
        scatter_kws = {"color": "black", "alpha": scatter_kws_custom_alpha, "s": scatter_plot_custom_size}
        if no_lm:
            ax = sns.scatterplot(x=x, y=y, **scatter_kws)
        else:
            ax = sns.regplot(x=x, y=y, scatter_kws=scatter_kws, line_kws={"color": "red"})
    # reset ymin, max, etc. based on xticks yticks
    if (xticks is not None) and (len(xticks) > 0):
        xmin = min(xmin, min(xticks))
        xmax = max(xmax, max(xticks))
    if (yticks is not None) and (len(yticks) > 0):
        ymin = min(ymin, min(yticks))
        ymax = max(ymax, max(yticks))
    xrange = xmax - xmin
    yrange = ymax - ymin
    if text_pos.split(" ")[0] == "top":
        text_y = ymin + yrange * .98
        if verticalalignment is None:
            verticalalignment = "top"
    else:
        text_y = ymin + yrange * .02
        if verticalalignment is None:
            verticalalignment = "bottom"
    if text_pos.split(" ")[1] == "left":
        text_x = xmin + xrange * .02
        if horizontalalignment is None:
            horizontalalignment = "left"
    else:
        text_x = xmax - xrange * .02
        if horizontalalignment is None:
            horizontalalignment = "right"
    if text_show:
        if no_lm and text_show_sample_size:
            if (group_name is not None) and (len(group_name) > 0):
                lm_text = f'{group_name}: '
            else:
                lm_text = ""
            lm_text += r'$\it{n}$' + f' = {len(x)}'
        elif (not no_lm) and text_show_sample_size:
            if (group_name is not None) and (len(group_name) > 0):
                lm_text = f'{group_name}: '
            else:
                lm_text = ""
            lm_text += r'$\it{n}$' + f' = {len(x)}' + '\n' + r"$\it{r}$" + f' = {r:.2f}'
            if confidence_level is not None:
                lm_text += f" ({r_lo:.2f}, {r_hi:.2f})"
        elif (not no_lm) and (not text_show_sample_size):
            lm_text = (r"$\it{r}$" + f' = {r:.2f}')
            if confidence_level is not None:
                lm_text += f" ({r_lo:.2f}, {r_hi:.2f})"
        else:
            lm_text = ""
            text_show = False
        if text_show:
            plt.text(text_x, text_y,
                     # '\n' + '$\it{slope} = $' + f'{slope:.1f}' + ', $\it{intercept} = $' + f'{intercept:.1f}' +
                     lm_text,
                     verticalalignment=verticalalignment,
                     horizontalalignment=horizontalalignment,
                     font=legend_font_params,
                     bbox=dict(facecolor='white', alpha=text_bbox_alpha) if text_bbox_alpha > 0 else None)
    if common_scale:
        plt.xlim(xymin, xymax)
        plt.ylim(xymin, xymax)
    if xticks is not None:
        plt.xticks(xticks)
    else:
        plt.locator_params(axis='x', nbins=n_xticks)
    xticks = ax.get_xticks()
    if (xticks % 1 == 0).all():
        xticks = xticks.astype(int)
    round_yticks = False
    if yticks is not None:
        plt.yticks(yticks)
    else:
        round_yticks = True
        plt.locator_params(axis='y', nbins=n_yticks)
    yticks = ax.get_yticks()
    if (yticks % 1 == 0).all():
        yticks = yticks.astype(int)
    elif round_yticks:
        # round intelligently
        n_decimalzeros = np.ceil(-np.log10((np.max(yticks) - np.min(yticks)) / 100))
        yticks = np.round(yticks, n_decimalzeros)
    ax.set_xticklabels(xticks, fontdict=ticks_font_params)
    ax.set_yticklabels(yticks, fontdict=ticks_font_params)
    plt.ylabel(ylabel, **labels_font_params)
    plt.xlabel(xlabel, **labels_font_params)
    if show_grid:
        plt.grid(color='#aaaaaa', linewidth=0.5)
    if suptitle is not None:
        plt.title(suptitle, loc='left', font=paneltitle_font_params)
    if console_print_pr:
        print(f"r={r:.2f}, p={p:.1g}")
    return ax


# quick function to do regression
def plot_box(x, y, hue, xlabel, ylabel, yticks=None, n_yticks: int = 5, showmeans: bool = True, suptitle: str = None,
             legend_loc: str = None, legend_off: bool = False,
             data=None, palette=None,
             show_grid: bool = True,
             subtype="boxplot"):
    # the actual box plot
    ax = None
    if subtype == "boxplot":
        ax = sns.boxplot(data=data, x=x, y=y, hue=hue,
                         palette=palette,
                         linewidth=boxplot_lw,
                         notch=False,
                         # saturation=1.,
                         **boxplot_props,
                         showmeans=showmeans)
    elif subtype == "violinplot":
        ax = sns.violinplot(data=data, x=x, y=y, hue=hue,
                            palette=palette,
                            linewidth=boxplot_lw,
                            # **boxplot_props,
                            # showmeans=showmeans,
                            )
    elif subtype == "precise_violinplot":
        ax = sns.violinplot(data=data, x=x, y=y, hue=hue,
                            palette=palette,
                            linewidth=boxplot_precise_lw,
                            # **boxplot_props,
                            # showmeans=showmeans,
                            )
    elif subtype == "precise_violinplot+stripplot":
        ax = sns.violinplot(data=data, x=x, y=y, hue=hue,
                            palette=palette,
                            linewidth=boxplot_precise_lw,
                            # **boxplot_props,
                            # showmeans=showmeans,
                            )
        sns.stripplot(data=data, x=x, y=y, hue=hue,
                      # linewidth=boxplot_lw,
                      # saturation=1.,
                      # **boxplot_props,
                      # showmeans=showmeans,
                      # showmeans=showmeans,
                      )
    elif subtype == "stripplot":
        ax = sns.stripplot(data=data, x=x, y=y, hue=hue,
                           palette=palette,
                           # linewidth=boxplot_lw,
                           # saturation=1.,
                           # **boxplot_props,
                           # showmeans=showmeans,
                           )
    elif subtype == "swarmplot":
        ax = sns.swarmplot(data=data, x=x, y=y, hue=hue,
                           palette=palette,
                           # linewidth=boxplot_lw,
                           # saturation=1.,
                           # **boxplot_props,
                           # showmeans=showmeans,
                           )
    else:
        assert False, f"Unknown {subtype=}"
    # ticks and labels
    if yticks is not None:
        plt.yticks(yticks)
    else:
        plt.locator_params(axis='y', nbins=n_yticks)
    yticks = ax.get_yticks()
    if (yticks % 1 == 0).all():
        yticks = yticks.astype(int)
    xticks = ax.get_xticks()
    if (data is not None) and (type(x) is str):
        if data[x].dtype.name == "category":
            xticks = [data[x].cat.categories[v] for v in xticks]
    elif type(x) is pd.Series:
        if x.dtype.name == "category":
            xticks = [x.cat.categories[v] for v in xticks]
    ax.set_xticklabels(xticks, fontdict=ticks_font_params)
    ax.set_yticklabels(yticks, fontdict=ticks_font_params)
    plt.ylabel(ylabel, **labels_font_params)
    plt.xlabel(xlabel, **labels_font_params)
    if suptitle is not None:
        plt.title(suptitle, loc='left', font=paneltitle_font_params)
    # grid
    if show_grid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='#aaaaaa', linewidth=0.5)
        # plt.grid(axis="y", color='#aaaaaa', linewidth=0.5)
    # legend
    if not legend_off:
        if legend_loc == "outside":
            ax.legend(prop=legend_font_params, bbox_to_anchor=(1.04, 0.5), loc="center left")
        else:
            ax.legend(prop=legend_font_params, loc=legend_loc)
    return ax


def plot_roc(y, y_, suptitle: str = None, display_n: bool = False, text_bbox_alpha: float = .0, confidence_level: float = None):
    if confidence_level is not None:
        # using confidenceinterval
        import confidenceinterval

        assert (confidence_level > 0) and (confidence_level < 1), f"Unexpected {confidence_level=}"

        t = np.concatenate([[np.Inf, ], np.sort(np.unique(y_))[::-1]])
        tpr, tpr_lo, tpr_hi, fpr, fpr_lo, fpr_hi = [], [], [], [], [], []
        for thresh in t:
            thresh_tpr, (thresh_tpr_lo, thresh_tpr_hi) = confidenceinterval.tpr_score(y, (y_ >= thresh) * 1, confidence_level=confidence_level)
            thresh_fpr, (thresh_fpr_lo, thresh_fpr_hi) = confidenceinterval.fpr_score(y, (y_ >= thresh) * 1, confidence_level=confidence_level)
            tpr.append(thresh_tpr)
            tpr_lo.append(thresh_tpr_lo)
            tpr_hi.append(thresh_tpr_hi)
            fpr.append(thresh_fpr)
            fpr_lo.append(thresh_fpr_lo)
            fpr_hi.append(thresh_fpr_hi)
        tpr, tpr_lo, tpr_hi, fpr, fpr_lo, fpr_hi = np.array(tpr), np.array(tpr_lo), np.array(tpr_hi), np.array(fpr), np.array(fpr_lo), np.array(fpr_hi)

        roc_auc, (roc_auc_lo, roc_auc_hi) = confidenceinterval.roc_auc_score(y, y_, confidence_level=confidence_level)

    else:
        # using sklearn
        from sklearn import metrics

        fpr, tpr, t = metrics.roc_curve(y, y_)
        tpr_lo, tpr_hi, fpr_lo, fpr_hi = None, None, None, None
        roc_auc = metrics.auc(fpr, tpr)
        roc_auc_lo, roc_auc_hi = None, None

    # confidenceinterval.roc_auc_score(y_true=balanced_df["Binary race"], y_pred=balanced_df[peak_method])

    # Plot curve
    plt.plot([0, 1], [0, 1], color='#aaaaaa', lw=2, linestyle='--')

    if confidence_level is not None:
        plt.fill_between(x=fpr, y1=tpr_lo, y2=tpr_hi, color='black', alpha=.1)
        plt.plot(fpr, tpr, color='black', lw=2, label=f'AUC={roc_auc:0.2f} ({roc_auc_lo:0.2f}-{roc_auc_hi:0.2f})')
    else:
        plt.plot(fpr, tpr, color='black', lw=2, label=f'AUC={roc_auc:0.2f}')

    if display_n:
        plt.text(.05, .95,
                 '$\it{n}$' + f' = {len(y)}',
                 verticalalignment="top",
                 horizontalalignment="left",
                 font=legend_font_params,
                 bbox=dict(facecolor='white', alpha=text_bbox_alpha) if text_bbox_alpha > 0 else None)

    # Set boundaries, labels, titles...
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])

    ax = plt.gca()
    xticks = np.round(100 * ax.get_xticks(), 0).astype(int)
    yticks = np.round(100 * ax.get_yticks(), 0).astype(int)
    ax.set_xticklabels(xticks, fontdict=ticks_font_params)
    ax.set_yticklabels(yticks, fontdict=ticks_font_params)

    plt.xlabel('False Positive Rate', **labels_font_params)
    plt.ylabel('True Positive Rate', **labels_font_params)
    if suptitle is not None:
        plt.title(suptitle, loc='left', font=paneltitle_font_params)
        # plt.title(suptitle)
    plt.legend(loc="lower right")
    plt.tight_layout()
    return ax


# quick function to do regression
def plot_paired_box(data, x, y, xlabel, ylabel, yticks=None, n_yticks: int = 7, showmeans: bool = False, suptitle: str = None,
                    legend_loc: str = None, legend_off: bool = False, palette=None, points_palette=None):
    # the actual box plot
    ax = sns.boxplot(data=data, x=x, y=y,
                     palette=palette,
                     linewidth=boxplot_lw,
                     notch=False,
                     # saturation=1.,
                     **boxplot_props,
                     showmeans=showmeans)
    scatter_kws = {"alpha": 1.0, "s": scatter_plot_size}
    sns.scatterplot(data=data, x=x, y=y, ax=ax, **scatter_kws, palette=points_palette, hue=x)
    x1, x2 = data[x].unique()
    for y1, y2 in zip(data[data[x] == x1][y].tolist(), data[data[x] == x2][y].tolist()):
        lcolor = "#ff0000" if y2 > y1 else "#0000ff"
        sns.lineplot(x=[x1, x2], y=[y1, y2], ax=ax, color=lcolor)
    # ticks and labels
    if yticks is not None:
        plt.yticks(yticks)
    else:
        plt.locator_params(axis='y', nbins=n_yticks)
    yticks = ax.get_yticks()
    if (yticks % 1 == 0).all():
        yticks = yticks.astype(int)
    xticks = ax.get_xticks()
    if (data is not None) and (type(x) is str):
        if data[x].dtype.name == "category":
            xticks = [data[x].cat.categories[v] for v in xticks]
    elif type(x) is pd.Series:
        if x.dtype.name == "category":
            xticks = [x.cat.categories[v] for v in xticks]
    ax.set_xticklabels(xticks, fontdict=ticks_font_params)
    ax.set_yticklabels(yticks, fontdict=ticks_font_params)
    plt.ylabel(ylabel, **labels_font_params)
    plt.xlabel(xlabel, **labels_font_params)
    if suptitle is not None:
        plt.title(suptitle, loc='left', font=paneltitle_font_params)
    # legend
    if not legend_off:
        if legend_loc == "outside":
            ax.legend(prop=legend_font_params, bbox_to_anchor=(1.04, 0.5), loc="center left")
        else:
            ax.legend(prop=legend_font_params, loc=legend_loc)
    else:
        ax.legend_.remove()
    return ax
