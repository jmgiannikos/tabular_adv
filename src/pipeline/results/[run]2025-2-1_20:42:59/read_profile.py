import pstats

stats = pstats.Stats("./time_profile")
stats.sort_stats("cumtime")
stats.print_stats()