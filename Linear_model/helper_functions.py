import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


"""
Function to plot data before and after the PCA

"""


def plot_PCA(pca, topic_str, experiments):
    X = experiments.values

    # X = np.log(X)
    # X = (X-X.mean(axis=0))/X.std(axis=0)
    eigenvectors = pca.components_

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(topic_str, size="xx-large")

    ### Plot before PCA
    ax[0].plot(X[:, 0], X[:, 1], "o")

    ax[0].grid()
    ax[0].tick_params(labelsize="xx-large")
    ax[0].set_xlabel(f"{experiments.columns[0]}", size="xx-large")
    ax[0].set_ylabel(f"{experiments.columns[1]}", size="xx-large")
    ax[0].set_title("Before PCA", size="xx-large")

    xmin, xmax = ax[0].get_xlim()

    x0 = np.linspace(0, xmax)
    y0 = x0 * eigenvectors[0][1] / eigenvectors[0][0]

    x1 = np.linspace(xmin, -1 * xmin)
    y1 = x1 * eigenvectors[1][1] / eigenvectors[1][0]

    ax[0].plot(x0, y0)
    ax[0].plot(x1, y1)

    ### Plot after PCA
    variances = pca.explained_variance_ratio_
    transformed_X = pca.transform(X)

    ax[1].plot(transformed_X[:, 0], transformed_X[:, 1], "o")

    ax[1].grid()
    ax[1].tick_params(labelsize="xx-large")
    ax[1].set_xlabel(f"PC 1 ({round(100*variances[0], 1)}%)", size="xx-large")
    ax[1].set_ylabel(f"PC 2 ({round(100*variances[1], 1)}%)", size="xx-large")
    ax[1].set_title("After PCA", size="xx-large")

    ax[1].axhline(color="k")
    ax[1].axvline(color="k")


"""
Function to return first eigenvector of the PCA for a given topic
"""


def get_first_component(topic, centrality, all_data, topics, PCA_path, component):
    topic_str = topics[topic]

    data = all_data.loc[(all_data.topic == topic_str) & (all_data.week == -1)]
    null = pd.read_csv(PCA_path + f"nodality_null_lbl{topic}.csv")

    experiments = (
        data[["username", centrality]]
        .set_index("username")
        .join(
            null[["username", centrality]].set_index("username"),
            lsuffix="_topic",
            rsuffix="_null",
        )
        .dropna(axis=0)
    )
    experiments = experiments[[centrality + "_null", centrality + "_topic"]]
    X = experiments.values
    X = np.log(X)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(X)

    return pca.components_[component]


"""
Function to get best centrality measure with respect to variances of PCA
"""


def get_best_centrality(topic, all_data, topics, centralities, PCA_path):
    topic_str = topics[topic]
    print(f"Topic: {topic_str}\n")

    data = all_data.loc[(all_data.topic == topic_str) & (all_data.week == -1)]
    null = pd.read_csv(PCA_path + f"nodality_null_lbl{topic}.csv")

    print(f"Length of data for topic {topic_str} network: {len(data)}")
    print(f"Length of data for null network: {len(null)}\n")

    for centrality in centralities:
        experiments = (
            data[["username", centrality]]
            .set_index("username")
            .join(
                null[["username", centrality]].set_index("username"),
                lsuffix="_topic",
                rsuffix="_null",
            )
            .dropna(axis=0)
        )
        experiments = experiments[[centrality + "_null", centrality + "_topic"]]

        X = np.log(experiments[experiments.columns[1:]])
        X = X.replace([-np.inf, np.inf], np.nan)
        X = X.values()

        X = experiments.values

        # X = np.log(X)
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        pca = PCA(n_components=2, svd_solver="full")
        pca.fit(X)

        variances = pca.explained_variance_ratio_

        print(f"Variance for centrality {centrality}: {variances}")
    print("\n")


"""
PCA pipeline to get two important set of actors
"""


def pipeline_PCA(centrality, topic, all_data, topics, PCA_path):
    topic_str = topics[topic]
    print(f"Centrality used: {centrality}\n")

    data = all_data.loc[(all_data.topic == topic_str) & (all_data.week == -1)]
    null = pd.read_csv(PCA_path + f"nodality_null_lbl{topic}.csv")

    print(f"Length of data for topic {topic_str} network: {len(data)}")
    print(f"Length of data for null network: {len(null)}\n")

    experiments = (
        data[["username", centrality]]
        .set_index("username")
        .join(
            null[["username", centrality]].set_index("username"),
            lsuffix="_topic",
            rsuffix="_null",
        )
        .dropna(axis=0)
    )
    experiments = experiments[[centrality + "_null", centrality + "_topic"]]
    X = experiments.values

    X = np.log(X)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    print(f"Number of nodes used for PCA analysis: {len(X)}\n")

    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(X)

    transformed_X = pca.transform(X)

    first_eigenvector = pca.components_[0]
    if first_eigenvector[0] >= first_eigenvector[1]:
        null_dim = 0
        topic_dim = 1

    if first_eigenvector[1] > first_eigenvector[0]:
        null_dim = 1
        topic_dim = 0

    authority = np.logical_and(
        transformed_X[:, null_dim] > 0, transformed_X[:, topic_dim] > 0
    )
    usernames_authority = experiments.loc[authority, :].index.values

    gained_authority = np.logical_and(
        transformed_X[:, null_dim] < 0, transformed_X[:, topic_dim] > 0
    )
    usernames_gained = experiments.loc[gained_authority, :].index.values

    print(f'Number of nodes with "popularity": {sum(authority)}')
    print(f'Number of nodes with "topic specificity": {sum(gained_authority)}')

    plot_PCA(pca, topic_str, experiments)

    return X, transformed_X, usernames_authority, usernames_gained


"""
MI Calculator
"""


def H(a):
    _, p = np.unique(a, axis=0, return_counts=True)
    p = p / np.sum(p)
    h = -1 * np.sum(p * np.log2(p))
    return h


def get_mi(a, b):
    mi = H(a) + H(b) - H(np.column_stack((a, b)))
    return mi


def get_cmi(a, b, c):
    cmi = get_mi(a, np.column_stack((b, c))) - get_mi(a, c)
    return cmi


def get_mi_diff_pop(X, Y, niter=100):
    N, T = np.shape(X)
    M, _ = np.shape(Y)

    te_xy = []
    te_yx = []

    for z in range(niter):
        selected_idx = np.random.choice(M, size=N, replace=False)
        y = Y[selected_idx]
        xy_temp = []
        yx_temp = []

        for i in range(1, T):
            txy = get_cmi(X[:, i - 1], y[:, i], y[:, i - 1])
            tyx = get_cmi(y[:, i - 1], X[:, i], X[:, i - 1])
            xy_temp.append(txy)
            yx_temp.append(tyx)
        te_xy.append(xy_temp)
        te_yx.append(yx_temp)
    te_xy = np.mean(te_xy, axis=0)
    te_yx = np.mean(te_yx, axis=0)
    return te_xy, te_yx


def get_weekly_activity(topic, path=""):
    journalists_filename = path + f"Journalist_activity_timeseries_lbl{topic}.csv"
    journalists_timeseries = pd.read_csv(journalists_filename)

    mp_filename = path + f"MP_activity_timeseries_lbl{topic}.csv"
    mp_timeseries = pd.read_csv(mp_filename)

    timeseries = pd.concat([journalists_timeseries, mp_timeseries]).reset_index(
        drop=True
    )
    timeseries = add_weekly_activity(timeseries)

    timeseries_daily = timeseries[timeseries.columns[:367]]
    columns_weekly = ["Screen_name"] + ["Week" + str(i) for i in range(52)]
    timeseries_weekly = timeseries[columns_weekly]
    idx_actives = timeseries_weekly[timeseries_weekly.columns[1:]].sum(axis=1) != 0
    timeseries_weekly = timeseries_weekly.loc[idx_actives, :].reset_index(drop=True)

    return timeseries_weekly


def add_weekly_activity(timeseries):
    timeseries_copy = timeseries.copy()
    min_day = 0
    max_day = 365

    day = min_day + 1
    number_week = 0

    while day < max_day:
        idx_min = day
        full_week = day + 7
        if full_week > max_day:
            full_week = max_day
        idx_max = full_week
        timeseries_copy[f"Week{number_week}"] = timeseries[
            timeseries.columns[idx_min:idx_max]
        ].sum(axis=1)

        day = full_week
        number_week += 1

    return timeseries_copy


def plot_TE_groups(topic, gained, authority, timeseries_weekly, topics, ymax, share):
    topic_str = topics[topic]
    merged = np.concatenate([gained, authority])

    groups = [authority, gained, merged]
    titles = ["Popular", "Topic specific", "Both"]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(topic_str, size="xx-large")

    len_ = len(groups)
    means = np.zeros(len_)
    stds = np.zeros(len_)
    mean_areas = np.zeros(len_)

    for i, arr in enumerate(groups):
        X = timeseries_weekly.loc[
            timeseries_weekly.Screen_name.isin(arr), timeseries_weekly.columns[1:]
        ].values
        Y = timeseries_weekly.loc[
            ~timeseries_weekly.Screen_name.isin(arr), timeseries_weekly.columns[1:]
        ].values

        te_xy, te_yx = get_mi_diff_pop(X, Y)

        if share:
            H_X = np.array([H(X[:, i]) for i in range(X.shape[1])])
            share_H_X = [
                100 * te_yx[i] / H_X[i + 1]
                if (te_yx[i] != 0) and (H_X[i + 1] != 0)
                else 0
                for i in range(len(te_yx))
            ]
            share_H_X = np.array(share_H_X)

            H_Y = np.array([H(Y[:, i]) for i in range(Y.shape[1])])
            share_H_Y = [
                100 * te_xy[i] / H_Y[i + 1]
                if (te_xy[i] != 0) and (H_Y[i + 1] != 0)
                else 0
                for i in range(len(te_xy))
            ]
            share_H_Y = np.array(share_H_Y)

            ax[i].plot(share_H_X, label="predicted info X")
            ax[i].plot(share_H_Y, label="predicted info Y")

            mean_ = round((share_H_Y - share_H_X).mean(), 2)
            std_ = round((share_H_Y - share_H_X).std(), 2)

            mean_area = round((share_H_X.sum() + share_H_Y.sum()) / 2, 2)

            means[i] = mean_
            stds[i] = std_
            mean_areas[i] = mean_area

            # print(f'X - {titles[i]} mean gain over Y : {mean_} pm {std_} and mean area: {mean_area}')
            ax[i].set_ylabel("share of H (%)", size="xx-large")
        else:
            ax[i].plot(te_xy, label="te_xy")
            ax[i].plot(te_yx, label="te_yx")
            ax[i].set_ylabel("TE (bits)", size="xx-large")

        ax[i].set_title(titles[i], size="xx-large")
        ax[i].legend()
        ax[i].grid()
        ax[i].set_ylim(ymax=ymax)
        ax[i].set_xlabel("week", size="xx-large")

    return means, stds, mean_areas


def plot_TE_groups_disaggregated(
    topic,
    gained,
    authority,
    timeseries_weekly,
    topics,
    labels_journalists,
    labels_mps,
    ymax,
    share,
):
    topic_str = topics[topic]

    journalists_authority = labels_journalists.loc[
        labels_journalists.username.isin(authority)
    ].username.values
    mps_authority = labels_mps.loc[labels_mps.username.isin(authority)].username.values

    journalists_gained = labels_journalists.loc[
        labels_journalists.username.isin(gained)
    ].username.values
    mps_gained = labels_mps.loc[labels_mps.username.isin(gained)].username.values

    groups = [[mps_authority, mps_gained], [journalists_authority, journalists_gained]]
    titles = [
        ["Popular MPs", "Topic spe. MPs"],
        ["Popular Journalists", "Topic spe. Journalists"],
    ]

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(topic_str, size="xx-large")

    means = np.zeros((2, 2))
    stds = np.zeros((2, 2))
    mean_areas = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            arr = groups[i][j]

            X = timeseries_weekly.loc[
                timeseries_weekly.Screen_name.isin(arr), timeseries_weekly.columns[1:]
            ].values
            Y = timeseries_weekly.loc[
                ~timeseries_weekly.Screen_name.isin(arr), timeseries_weekly.columns[1:]
            ].values

            te_xy, te_yx = get_mi_diff_pop(X, Y)

            if share:
                H_X = np.array([H(X[:, i]) for i in range(X.shape[1])])
                share_H_X = [
                    100 * te_yx[i] / H_X[i + 1]
                    if (te_yx[i] != 0) and (H_X[i + 1] != 0)
                    else 0
                    for i in range(len(te_yx))
                ]
                share_H_X = np.array(share_H_X)

                H_Y = np.array([H(Y[:, i]) for i in range(Y.shape[1])])
                share_H_Y = [
                    100 * te_xy[i] / H_Y[i + 1]
                    if (te_xy[i] != 0) and (H_Y[i + 1] != 0)
                    else 0
                    for i in range(len(te_xy))
                ]
                share_H_Y = np.array(share_H_Y)

                ax[i, j].plot(share_H_X, label="predicted info X")
                ax[i, j].plot(share_H_Y, label="predicted info Y")

                mean_ = round((share_H_Y - share_H_X).mean(), 2)
                std_ = round((share_H_Y - share_H_X).std(), 2)
                mean_area = round((share_H_X.sum() + share_H_Y.sum()) / 2, 2)

                means[i, j] = mean_
                stds[i, j] = std_
                mean_areas[i, j] = mean_area

                # print(f'X - {titles[i][j]} mean gain over Y : {mean_} pm {std_} and mean area: {mean_area}')
                ax[i, j].set_ylabel("share of H (%)", size="xx-large")
            else:
                ax[i, j].plot(te_xy, label="te_xy")
                ax[i, j].plot(te_yx, label="te_yx")
                ax[i, j].set_ylabel("TE (bits)", size="xx-large")

            ax[i, j].set_title(titles[i][j], size="xx-large")
            ax[i, j].legend()
            ax[i, j].grid()
            ax[i, j].set_ylim(ymax=ymax)
            ax[i, j].set_xlabel("week", size="xx-large")

    return means, stds, mean_areas


def plot_TE_2groups(
    topic,
    gained,
    authority,
    timeseries_weekly,
    topics,
    labels_journalists,
    labels_mps,
    ymax,
    share,
):
    topic_str = topics[topic]

    journalists_authority = labels_journalists.loc[
        labels_journalists.username.isin(authority)
    ].username.values
    mps_authority = labels_mps.loc[labels_mps.username.isin(authority)].username.values

    journalists_gained = labels_journalists.loc[
        labels_journalists.username.isin(gained)
    ].username.values
    mps_gained = labels_mps.loc[labels_mps.username.isin(gained)].username.values

    journalists = np.concatenate([journalists_authority, journalists_gained])
    mps = np.concatenate([mps_authority, mps_gained])

    groups = [mps, journalists]
    titles = ["Nodal MPs", "Nodal Journalists"]

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(topic_str, size="xx-large")

    len_ = len(groups)
    means = np.zeros(len_)
    stds = np.zeros(len_)
    mean_areas = np.zeros(len_)

    for i, arr in enumerate(groups):
        X = timeseries_weekly.loc[
            timeseries_weekly.Screen_name.isin(arr), timeseries_weekly.columns[1:]
        ].values
        Y = timeseries_weekly.loc[
            ~timeseries_weekly.Screen_name.isin(arr), timeseries_weekly.columns[1:]
        ].values

        te_xy, te_yx = get_mi_diff_pop(X, Y)

        if share:
            H_X = np.array([H(X[:, i]) for i in range(X.shape[1])])
            share_H_X = [
                100 * te_yx[i] / H_X[i + 1]
                if (te_yx[i] != 0) and (H_X[i + 1] != 0)
                else 0
                for i in range(len(te_yx))
            ]
            share_H_X = np.array(share_H_X)

            H_Y = np.array([H(Y[:, i]) for i in range(Y.shape[1])])
            share_H_Y = [
                100 * te_xy[i] / H_Y[i + 1]
                if (te_xy[i] != 0) and (H_Y[i + 1] != 0)
                else 0
                for i in range(len(te_xy))
            ]
            share_H_Y = np.array(share_H_Y)

            ax[i].plot(share_H_X, label="predicted info X")
            ax[i].plot(share_H_Y, label="predicted info Y")

            mean_ = round((share_H_Y - share_H_X).mean(), 2)
            std_ = round((share_H_Y - share_H_X).std(), 2)
            mean_area = round((share_H_X.sum() + share_H_Y.sum()) / 2, 2)

            means[i] = mean_
            stds[i] = std_
            mean_areas[i] = mean_area

            # print(f'X - {titles[i]} mean gain over Y : {mean_} pm {std_} and mean area: {mean_area}')
            ax[i].set_ylabel("share of H (%)", size="xx-large")
        else:
            ax[i].plot(te_xy, label="te_xy")
            ax[i].plot(te_yx, label="te_yx")
            ax[i].set_ylabel("TE (bits)", size="xx-large")

        ax[i].set_title(titles[i], size="xx-large")
        ax[i].legend()
        ax[i].grid()
        ax[i].set_ylim(ymax=ymax)
        ax[i].set_xlabel("week", size="xx-large")

    return means, stds, mean_areas


def get_list_features(best, measures, id_):
    l = best.loc[best.id == id_, "features"].values[0]
    list_ = list()
    for measure in measures:
        if measure in l:
            list_.append(measure)
    return list_


def get_lists_of_list_features(best, measures):
    list_of_lists = list()
    for id_ in best.id.values:
        list_ = get_list_features(best, measures, id_)
        list_of_lists.append(list_)
    return list_of_lists


def transform_common_nodes_to_list(best):
    list_of_lists = list()
    for id_ in best.id.values:
        string = best.loc[best.id == id_, "common_nodes"].values[0]
        l = string.split("'")[1::2]
        len_ = best.loc[best.id == id_, "no_of_common_nodes"].values[0]
        if (type(l) == list) and (len(l) == len_):
            list_of_lists.append(l)
        else:
            list_of_lists.append(string)
    best["common_nodes"] = list_of_lists

    return best


def get_second_group(best):
    column_name = "second_common_nodes"
    list_of_lists = list()
    cardinalities = list()
    if column_name not in best.columns:
        for id_ in best.id.values:
            nodes = pd.read_csv(path + feature_name + str(id_) + "/" + non_nodal_name)[
                "0"
            ].to_list()
            len_nodes = len(nodes)
            other_nodes = best.loc[best.id == id_, "common_nodes"].values
            if nodes[0] in other_nodes:
                nodes = pd.read_csv(
                    path + feature_name + str(id_) + "/" + inherent_name
                )["0"].to_list()
                len_nodes = len(nodes)

            list_of_lists.append(nodes)
            cardinalities.append(len_nodes)

        best[column_name] = list_of_lists
        best["no_of_" + column_name] = cardinalities

        return best
    else:
        print("second group of common nodes is already in best DataFrame")
        return best


def are_features_the_same_test(best):
    are_the_same = list()

    for id_ in best.id.values:
        df = pd.read_csv(
            path_features + str(id_) + "/" + feature_list, index_col=0
        ).rename(columns={"0": "features"})
        df = df.loc[: len(df) / 2 - 1, "features"].values

        list_ = np.array(best.loc[best.id == id_, "features"].values[0])

        b = bool(np.floor(sum(np.equal(np.sort(list_), np.sort(df))) / len(df)))
        are_the_same.append(b)

    are_they = bool(np.floor(sum(are_the_same)))
    return are_they


def get_df_counts(best, measures):
    counts = np.zeros(len(measures))
    for list_ in best.features.values:
        for i, measure in enumerate(measures):
            if measure in list_:
                counts[i] += 1
    df_counts = pd.DataFrame()
    df_counts["measures"] = measures
    df_counts["counts"] = counts

    return df_counts


def get_mi_diff_pop_1(X, Y, niter=100):
    N, T = np.shape(X)
    M, _ = np.shape(Y)

    te_xy = []

    for z in range(niter):
        selected_idx = np.random.choice(M, size=N, replace=False)
        y = Y[selected_idx]
        xy_temp = []

        for i in range(1, T):
            txy = get_cmi(X[:, i - 1], y[:, i], y[:, i - 1])
            xy_temp.append(txy)
        te_xy.append(xy_temp)
    te_xy = np.mean(te_xy, axis=0)
    return te_xy


def get_share_H_Y(l_x, l_y, timeseries_weekly):
    X = timeseries_weekly.loc[
        timeseries_weekly.Screen_name.isin(l_x), timeseries_weekly.columns[1:]
    ].values
    Y = timeseries_weekly.loc[
        timeseries_weekly.Screen_name.isin(l_y), timeseries_weekly.columns[1:]
    ].values

    te_xy = get_mi_diff_pop_1(X, Y)

    H_Y = np.array([H(Y[:, i]) for i in range(Y.shape[1])])
    share_H_Y = [
        100 * te_xy[i] / H_Y[i + 1] if (te_xy[i] != 0) and (H_Y[i + 1] != 0) else 0
        for i in range(len(te_xy))
    ]
    share_H_Y = np.array(share_H_Y)

    return share_H_Y


def get_share_H_exclusive(l_x, timeseries_weekly):
    X = timeseries_weekly.loc[
        timeseries_weekly.Screen_name.isin(l_x), timeseries_weekly.columns[1:]
    ].values
    Y = timeseries_weekly.loc[
        ~timeseries_weekly.Screen_name.isin(l_x), timeseries_weekly.columns[1:]
    ].values

    te_xy, te_yx = get_mi_diff_pop(X, Y)

    H_X = np.array([H(X[:, i]) for i in range(X.shape[1])])
    share_H_X = [
        100 * te_yx[i] / H_X[i + 1] if (te_yx[i] != 0) and (H_X[i + 1] != 0) else 0
        for i in range(len(te_yx))
    ]
    share_H_X = np.array(share_H_X)

    H_Y = np.array([H(Y[:, i]) for i in range(Y.shape[1])])
    share_H_Y = [
        100 * te_xy[i] / H_Y[i + 1] if (te_xy[i] != 0) and (H_Y[i + 1] != 0) else 0
        for i in range(len(te_xy))
    ]
    share_H_Y = np.array(share_H_Y)

    return share_H_X, share_H_Y


def get_all_active_nodes(combinations_to_try):
    active_nodes = pd.DataFrame()

    for id_ in combinations_to_try.id.values:
        inherent = combinations_to_try.loc[
            combinations_to_try.id == id_, "inherent"
        ].values[0]
        non_nodal = combinations_to_try.loc[
            combinations_to_try.id == id_, "non_nodal"
        ].values[0]

        for topic in topics:
            for cluster in range(3):
                list_topic = pd.read_csv(
                    path + feature_name + f"{id_}/{topic}_{cluster}.csv"
                ).Node.to_list()

                len_inter_inherent = len(set(inherent).intersection(set(list_topic)))
                len_inter_non_nodal = len(set(non_nodal).intersection(set(list_topic)))

                if (len_inter_non_nodal == 0) and (len_inter_inherent == 0):
                    active_nodes[f"{topic}_{id_}"] = [list_topic]
                    active_nodes[f"len_{topic}_{id_}"] = len(list_topic)
    return active_nodes


def get_three_clusters(id_, combinations_to_try):
    three_clusters = pd.DataFrame()

    inherent = combinations_to_try.loc[
        combinations_to_try.id == id_, "inherent"
    ].values[0]
    non_nodal = combinations_to_try.loc[
        combinations_to_try.id == id_, "non_nodal"
    ].values[0]

    for topic in topics:
        for cluster in range(3):
            list_topic = pd.read_csv(
                path + feature_name + f"{id_}/{topic}_{cluster}.csv"
            ).Node.to_list()

            len_inter_inherent = len(set(inherent).intersection(set(list_topic)))
            len_inter_non_nodal = len(set(non_nodal).intersection(set(list_topic)))

            if (len_inter_non_nodal == 0) and (len_inter_inherent == 0):
                three_clusters[f"{topic}_active"] = [list_topic]
            elif len_inter_inherent != 0:
                three_clusters[f"{topic}_inherent"] = [list_topic]
            elif len_inter_non_nodal != 0:
                three_clusters[f"{topic}_non_nodal"] = [list_topic]
    return three_clusters


def get_four_clusters(id_, combinations_to_try):
    four_clusters = pd.DataFrame()

    inherent = combinations_to_try.loc[
        combinations_to_try.id == id_, "inherent"
    ].values[0]
    non_nodal = combinations_to_try.loc[
        combinations_to_try.id == id_, "non_nodal"
    ].values[0]

    four_clusters["inherent"] = [inherent]

    for topic in topics:
        for cluster in range(3):
            list_topic = pd.read_csv(
                path + feature_name + f"{id_}/{topic}_{cluster}.csv"
            ).Node.to_list()

            len_inter_inherent = len(set(inherent).intersection(set(list_topic)))
            len_inter_non_nodal = len(set(non_nodal).intersection(set(list_topic)))

            if (len_inter_non_nodal == 0) and (len_inter_inherent == 0):
                four_clusters[f"{topic}_funnel"] = [list_topic]
            elif len_inter_inherent != 0:
                four_clusters[f"{topic}_active"] = [
                    [node for node in list_topic if node not in inherent]
                ]
            elif len_inter_non_nodal != 0:
                four_clusters[f"{topic}_non_nodal"] = [list_topic]
    return four_clusters
