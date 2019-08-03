import json
import os
import logging
import inspect
import sys
from collections import defaultdict
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

DATEFORMAT = "%Y-%m-%d"
MESSAGE_DATEFORMAT = "%a, %d %b %Y %H:%M:%S %Z"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

COLORS = {"pass":"red", "like":"green", "match":"dimgray", "messages": "blue"}
BAR_WIDTH = .8
BAR_ALPHA = .4
SHOW_PLOTS = False

LOG = logging.getLogger(__name__)


# General ----------------------------------------------------------------------


def load_mydata(folder_path):
    with open(os.path.join(folder_path, "data.json")) as f:
        return json.loads(f.read())


def general_info(data):
    app_opens = data["Usage"]["app_opens"]
    min_date = min(v for v in app_opens.keys())
    max_date = max(v for v in app_opens.keys())
    used = len([v for v in app_opens.values() if v>0])

    delta = datetime.strptime(max_date, DATEFORMAT) - datetime.strptime(min_date, DATEFORMAT)

    LOG.info("--- General Info ---")
    LOG.info(f"Start Date: {min_date}")
    LOG.info(f"End Date:   {max_date}")
    LOG.info(f"Days:       {delta.days}")
    LOG.info(f"  - used:   {used}")
    LOG.info(f"  - unused: {delta.days - used}")
    LOG.info(f"Av. app-open per usage-day: {np.mean([int(v) for v in app_opens.values() if v > 0]):.1f}")
    LOG.info("")


# Messages ---------------------------------------------------------------------


def message_statistics(path, data):
    sent = data["Usage"]["messages_sent"]
    received = data["Usage"]["messages_received"]
    messages_total(sent, received)
    messages_monthly(path, sent, received)
    messages_weekday(path, sent, received)


def messages_total(sent, received):
    total_sent = sum(v for v in sent.values())
    total_received = sum(v for v in received.values())
    LOG.info("--- Message Statistics ---")
    LOG.info(f"Send:     {total_sent:,d}".replace(",", "'"))
    LOG.info(f"Received: {total_received:,d}".replace(",", "'"))
    LOG.info("")


def messages_monthly(path, sent, received):
    id_fun = lambda x: x[0:7]  # maps year and month
    sent_map = _split_tuples(sorted(_get_mapped_sum(id_fun, sent).items()))
    received_map = _split_tuples(sorted(_get_mapped_sum(id_fun, received).items()))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    ax.bar(sent_map[0], sent_map[1], BAR_WIDTH, label="Sent",
           color=to_rgba(COLORS["like"], BAR_ALPHA))
    ax.bar(received_map[0], received_map[1], BAR_WIDTH, bottom=sent_map[1], label="Received",
           color=to_rgba(COLORS["pass"], BAR_ALPHA))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=75, ha="center")
    ax.set_ylabel("Messages")
    ax.legend()

    ax1 = ax.twinx()
    ax1.plot(sent_map[0], np.array(received_map[1]) / np.array(sent_map[1]) * 100, color=COLORS["match"])
    ax1.set_ylabel("Received/Sent [%]", color=COLORS["match"])
    ax1.tick_params(axis='y', labelcolor=COLORS["match"])

    ax.set_title("Messages per Month")
    fig.tight_layout()
    _save_fig(path, fig)


def messages_weekday(path, sent, received):
    id_fun = lambda x: datetime.strptime(x, DATEFORMAT).weekday()  # maps to weekday
    sent_map = _split_tuples(sorted(_get_mapped_sum(id_fun, sent).items()))
    received_map = _split_tuples(sorted(_get_mapped_sum(id_fun, received).items()))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    ax.bar(sent_map[0], sent_map[1], BAR_WIDTH, label="Sent",
           color=to_rgba(COLORS["like"], BAR_ALPHA))
    ax.bar(received_map[0], received_map[1], BAR_WIDTH, bottom=sent_map[1], label="Received",
           color=to_rgba(COLORS["pass"], BAR_ALPHA))
    ax.xaxis.set_ticklabels([""] + WEEKDAYS)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=75, ha="center")
    ax.set_ylabel("Messages")
    ax.legend()

    ax1 = ax.twinx()
    ax1.plot(sent_map[0], np.array(received_map[1]) / np.array(sent_map[1]) * 100, color=COLORS["match"])
    ax1.set_ylabel("Received/Sent [%]", color=COLORS["match"])
    ax1.tick_params(axis='y', labelcolor=COLORS["match"])

    ax.set_title("Messages per Weekday")
    fig.tight_layout()
    _save_fig(path, fig)


def message_loyality(path, data):
    matches = list(reversed(data["Messages"]))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    for idx in range(len(matches)):
        messages = matches[idx]["messages"]
        dates = [datetime.strptime(m["sent_date"], MESSAGE_DATEFORMAT) for m in messages]
        ax.plot(dates, [idx] * len(dates), "-o", c=COLORS["messages"])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=75, ha="center")
    ax.set_ylabel("Match #")
    ax.set_title("Message Loyality")
    _save_fig(path, fig)


# Swipes -----------------------------------------------------------------------


def swipe_statistics(path, data):
    likes = data["Usage"]["swipes_likes"]
    passes = data["Usage"]["swipes_passes"]
    matches = data["Usage"]["matches"]
    swipes_total(likes, passes, matches)
    swipes_monthly(path, likes, passes, matches)
    swipes_monthly_relative(path, likes, passes, matches)
    swipes_weekday(path, likes, passes, matches)
    swipes_weekday_relative(path, likes, passes, matches)


def swipes_total(likes, passes, matches):
    total_likes = sum(v for v in likes.values())
    total_passes = sum(v for v in passes.values())
    total_swipes = total_likes + total_passes
    total_matches = sum(v for v in matches.values())
    LOG.info("--- Swipe Statistics ---")
    LOG.info(f"Total Swipes: {total_swipes:,d}".replace(",", "'"))
    LOG.info(f"- Passes: {total_passes:,d} ({total_passes/total_swipes*100:.1f}%)".replace(",", "'"))
    LOG.info(f"- Likes: {total_likes:,d} ({total_likes/total_swipes*100:.1f}%)".replace(",", "'"))
    LOG.info(f"- Matches: {total_matches:d} ({total_matches/total_swipes*100:.1f}% of swipes".replace(",", "'") +
             f", {total_matches/total_likes*100:.1f}% of likes)")
    LOG.info("")


def swipes_monthly(path, likes, passes, matches):
    id_fun = lambda x: x[0:7]  # maps year and month
    like_map = _split_tuples(sorted(_get_mapped_sum(id_fun, likes).items()))
    pass_map = _split_tuples(sorted(_get_mapped_sum(id_fun, passes).items()))
    match_map = _split_tuples(sorted(_get_mapped_sum(id_fun, matches).items()))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    ax.bar(like_map[0], like_map[1], BAR_WIDTH, label="Like",
           color=to_rgba(COLORS["like"], BAR_ALPHA))
    ax.bar(pass_map[0], pass_map[1], BAR_WIDTH, bottom=like_map[1], label="Pass",
           color=to_rgba(COLORS["pass"], BAR_ALPHA))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=75, ha="center")
    ax.set_ylabel("Swipes")
    ax.legend()

    ax1 = ax.twinx()
    ax1.plot(match_map[0], match_map[1], color=COLORS["match"])
    ax1.set_ylabel("Matches", color=COLORS["match"])
    ax1.tick_params(axis='y', labelcolor=COLORS["match"])

    ax.set_title("Swipes per Month")
    fig.tight_layout()
    _save_fig(path, fig)


def swipes_monthly_relative(path, likes, passes, matches):
    id_fun = lambda x: x[0:7]  # maps year and month
    like_map = _split_tuples(sorted(_get_mapped_sum(id_fun, likes).items()))
    pass_map = _split_tuples(sorted(_get_mapped_sum(id_fun, passes).items()))
    match_map = _split_tuples(sorted(_get_mapped_sum(id_fun, matches).items()))
    total = np.array(like_map[1]) + np.array(pass_map[1])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    ax.plot(like_map[0], np.array(like_map[1]) / total * 100,
            color=COLORS["like"], label="Likes/Swipes")
    ax.plot(match_map[0], np.array(match_map[1]) / np.array(like_map[1]) * 100,
            color=COLORS["match"], label="Matches/Likes")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=75, ha="center")
    ax.set_ylabel("Swipes [%]")
    ax.legend()

    ax.set_title("Swipes per Month")
    fig.tight_layout()
    _save_fig(path, fig)


def swipes_weekday(path, likes, passes, matches):
    id_fun = lambda x: datetime.strptime(x, DATEFORMAT).weekday()  # maps to weekday
    like_map = _split_tuples(sorted(_get_mapped_sum(id_fun, likes).items()))
    pass_map = _split_tuples(sorted(_get_mapped_sum(id_fun, passes).items()))
    match_map = _split_tuples(sorted(_get_mapped_sum(id_fun, matches).items()))

    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    ax.bar(like_map[0], like_map[1], BAR_WIDTH, label="Like",
           color=to_rgba(COLORS["like"], BAR_ALPHA))
    ax.bar(pass_map[0], pass_map[1], BAR_WIDTH, bottom=like_map[1], label="Pass",
           color=to_rgba(COLORS["pass"], BAR_ALPHA))
    ax.xaxis.set_ticklabels([""] + WEEKDAYS)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=75, ha="center")
    ax.set_ylabel("Swipes")
    ax.legend()

    ax1 = ax.twinx()
    ax1.plot(match_map[0], match_map[1], color=COLORS["match"])
    ax1.set_ylabel("Matches", color=COLORS["match"])
    ax1.tick_params(axis='y', labelcolor=COLORS["match"])

    ax.set_title("Swipes per Weekday")
    fig.tight_layout()
    _save_fig(path, fig)


def swipes_weekday_relative(path, likes, passes, matches):
    id_fun = lambda x: datetime.strptime(x, DATEFORMAT).weekday()  # maps to weekday
    like_map = _split_tuples(sorted(_get_mapped_sum(id_fun, likes).items()))
    pass_map = _split_tuples(sorted(_get_mapped_sum(id_fun, passes).items()))
    match_map = _split_tuples(sorted(_get_mapped_sum(id_fun, matches).items()))
    total = np.array(like_map[1]) + np.array(pass_map[1])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    ax.plot(like_map[0], np.array(like_map[1]) / total * 100,
            color=COLORS["like"], label="Likes/Swipes")
    ax.plot(match_map[0], np.array(match_map[1]) / np.array(like_map[1]) * 100,
            color=COLORS["match"], label="Matches/Likes")
    ax.xaxis.set_ticklabels([""] + WEEKDAYS)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=75, ha="center")
    ax.set_ylabel("Swipes [%]")
    ax.legend()

    ax.set_title("Swipes per Weekday")
    fig.tight_layout()
    _save_fig(path, fig)

# Helper -----------------------------------------------------------------------


def _get_mapped_sum(map, data):
    mapped = defaultdict(int)
    for k in data.keys():
        mapped[map(k)] += int(data[k])
    return mapped


def _split_tuples(tuple_list):
    return [t[0] for t in tuple_list], [t[1] for t in tuple_list]


def _save_fig(path, fig):
    name = inspect.stack()[1][3]
    fig.savefig(os.path.join(path, f"{name}.png"))
    if SHOW_PLOTS:
        plt.show()


# Main -------------------------------------------------------------------------


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Please specify Tinder Data Folder as first argument")
    path = sys.argv[1]
    logging.basicConfig(format="%(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(path, "logfile.txt"), mode="w"),
                                  logging.StreamHandler()],)
    data = load_mydata(path)
    general_info(data)
    swipe_statistics(path, data)
    message_statistics(path, data)
    message_loyality(path, data)
