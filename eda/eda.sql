-- Get table schema information for all tables in the hepatits dataset
SELECT 
  table_name, column_name, data_type, is_nullable
FROM 
  `aise3010finalproject.hepatits.INFORMATION_SCHEMA.COLUMNS`
ORDER BY 
  table_name, ordinal_position;

-- Count rows in each table
SELECT 'Bio' as table_name, COUNT(*) as row_count FROM `aise3010finalproject.hepatits.Bio` UNION ALL
SELECT 'dispat' as table_name, COUNT(*) as row_count FROM `aise3010finalproject.hepatits.dispat` UNION ALL
SELECT 'indis' as table_name, COUNT(*) as row_count FROM `aise3010finalproject.hepatits.indis` UNION ALL
SELECT 'inf' as table_name, COUNT(*) as row_count FROM `aise3010finalproject.hepatits.inf` UNION ALL
SELECT 'rel11' as table_name, COUNT(*) as row_count FROM `aise3010finalproject.hepatits.rel11` UNION ALL
SELECT 'rel12' as table_name, COUNT(*) as row_count FROM `aise3010finalproject.hepatits.rel12` UNION ALL
SELECT 'rel13' as table_name, COUNT(*) as row_count FROM `aise3010finalproject.hepatits.rel13`
ORDER BY 
  table_name;

-- Check for null values in important columns across tables
SELECT 'Bio - fibros' as column_check, SUM(CASE WHEN fibros IS NULL THEN 1 ELSE 0 END) as null_count FROM `aise3010finalproject.hepatits.Bio` UNION ALL
SELECT 'Bio - activity' as column_check, SUM(CASE WHEN activity IS NULL THEN 1 ELSE 0 END) as null_count FROM `aise3010finalproject.hepatits.Bio` UNION ALL
SELECT 'dispat - sex' as column_check, SUM(CASE WHEN sex IS NULL THEN 1 ELSE 0 END) as null_count FROM `aise3010finalproject.hepatits.dispat` UNION ALL
SELECT 'dispat - age' as column_check, SUM(CASE WHEN age IS NULL THEN 1 ELSE 0 END) as null_count FROM `aise3010finalproject.hepatits.dispat` UNION ALL
SELECT 'indis - got' as column_check, SUM(CASE WHEN got IS NULL THEN 1 ELSE 0 END) as null_count FROM `aise3010finalproject.hepatits.indis` UNION ALL
SELECT 'inf - dur' as column_check, SUM(CASE WHEN dur IS NULL THEN 1 ELSE 0 END) as null_count FROM `aise3010finalproject.hepatits.inf`
ORDER BY 
  column_check;

-- Distribution of categorical variables in dispat table
SELECT 
  sex, COUNT(*) as count
FROM 
  `aise3010finalproject.hepatits.dispat`
GROUP BY 
  sex
ORDER BY 
  count DESC;

SELECT 
  Type, COUNT(*) as count
FROM 
  `aise3010finalproject.hepatits.dispat`
GROUP BY 
  Type
ORDER BY 
  count DESC;

-- Check relationship integrity
-- Check if all b_id in Bio exist in rel11
SELECT
  COUNT(*) as orphaned_bio_records
FROM
  `aise3010finalproject.hepatits.Bio` b
LEFT JOIN
  `aise3010finalproject.hepatits.rel11` r
ON
  b.b_id = r.b_id
WHERE
  r.b_id IS NULL;

-- Check if all m_id in dispat exist in relevant relation tables
SELECT
  COUNT(*) as orphaned_dispat_records
FROM
  `aise3010finalproject.hepatits.dispat` d
LEFT JOIN
  `aise3010finalproject.hepatits.rel11` r11
ON
  d.m_id = r11.m_id
LEFT JOIN
  `aise3010finalproject.hepatits.rel12` r12
ON
  d.m_id = r12.m_id
LEFT JOIN
  `aise3010finalproject.hepatits.rel13` r13
ON
  d.m_id = r13.m_id
WHERE
  r11.m_id IS NULL AND r12.m_id IS NULL AND r13.m_id IS NULL;
