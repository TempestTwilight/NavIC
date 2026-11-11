
-- Audit sessions by start date and device model
SELECT 
    manufacturer || ' ' || model as device_name,
    COALESCE(named_day, '2026-06-04 (Orientation Campaign)') as recording_day,
    COUNT(*) as session_count
FROM read_csv_auto('/home/tempest/Documents/ISRO_Project/gnss-eda/Stats/scratch/sessions_meta.csv')
GROUP BY device_name, recording_day
ORDER BY recording_day, device_name;
