-- 001_stage1_smoke.sql — verify stage-1 lines cache against grep on one file
-- COUNT(*) and Raw-count must equal `grep -c '^[^#]'` and `grep -c '^Raw,'`
SELECT count(*) AS n_lines,
       count(*) FILTER (split_part(line, ',', 1) = 'Raw') AS n_raw
FROM read_csv('data/Raw_Logs/45_back_vertical.txt',
              auto_detect=false, header=false, comment='#', delim=',',
              columns={'line':'VARCHAR'}, strict_mode=false)
WHERE line <> '';
-- Result: 745344 | 57853 (matches grep exactly)
