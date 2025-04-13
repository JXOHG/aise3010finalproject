-- Create a clean version of the hepatitis dataset with proper data cleaning
CREATE OR REPLACE TABLE `aise3010finalproject.hepatits.clean_hepatitis_data` AS

WITH 
-- First calculate mean values for numerical columns to use for imputation
numerical_means AS (
  SELECT
    AVG(i.got) AS mean_got,
    AVG(i.gpt) AS mean_gpt,
    AVG(i.alb) AS mean_alb,
    AVG(i.tbil) AS mean_tbil,
    AVG(i.dbil) AS mean_dbil,
    AVG(i.che) AS mean_che,
    AVG(i.ttt) AS mean_ttt,
    AVG(i.ztt) AS mean_ztt,
    AVG(i.tcho) AS mean_tcho,
    AVG(i.tp) AS mean_tp,
    AVG(CAST(d.age AS FLOAT64)) AS mean_age,
    AVG(CAST(b.fibros AS FLOAT64)) AS mean_fibros,
    AVG(CAST(b.activity AS FLOAT64)) AS mean_activity
  FROM
    `aise3010finalproject.hepatits.indis` AS i,
    `aise3010finalproject.hepatits.dispat` AS d,
    `aise3010finalproject.hepatits.Bio` AS b
),

-- Join all required tables
joined_data AS (
  SELECT
    d.m_id,
    CAST(d.sex AS STRING) AS sex,
    CAST(d.age AS FLOAT64) AS age,
    CAST(d.Type AS STRING) AS Type,
    CAST(b.fibros AS FLOAT64) AS fibros,
    CAST(b.activity AS FLOAT64) AS activity,
    CAST(i.got AS FLOAT64) AS got,
    CAST(i.gpt AS FLOAT64) AS gpt,
    CAST(i.alb AS FLOAT64) AS alb,
    CAST(i.tbil AS FLOAT64) AS tbil,
    CAST(i.dbil AS FLOAT64) AS dbil,
    CAST(i.che AS FLOAT64) AS che,
    CAST(i.ttt AS FLOAT64) AS ttt,
    CAST(i.ztt AS FLOAT64) AS ztt,
    CAST(i.tcho AS FLOAT64) AS tcho,
    CAST(i.tp AS FLOAT64) AS tp,
    -- Create a row number to identify duplicates
    ROW_NUMBER() OVER (
      PARTITION BY d.m_id 
      ORDER BY d.m_id
    ) AS row_num
  FROM
    `aise3010finalproject.hepatits.indis` AS i
  JOIN
    `aise3010finalproject.hepatits.rel12` AS c
    ON i.in_id = c.in_id
  JOIN
    `aise3010finalproject.hepatits.dispat` AS d
    ON d.m_id = c.m_id
  JOIN
    `aise3010finalproject.hepatits.rel11` AS a
    ON a.m_id = d.m_id
  JOIN
    `aise3010finalproject.hepatits.Bio` AS b
    ON b.b_id = a.b_id
  WHERE
    d.Type IS NOT NULL
)

-- Select final cleaned dataset with imputed values
SELECT
  j.m_id,
  -- Handle categorical missing values
  COALESCE(j.sex, 'Unknown') AS sex,
  -- Convert sex to standard format
  CASE 
    WHEN LOWER(j.sex) IN ('m', 'male') THEN 'M'
    WHEN LOWER(j.sex) IN ('f', 'female') THEN 'F'
    ELSE COALESCE(j.sex, 'Unknown')
  END AS sex_standardized,
  
  -- Impute numerical missing values with mean
  COALESCE(j.age, m.mean_age) AS age,
  COALESCE(j.fibros, m.mean_fibros) AS fibros,
  COALESCE(j.activity, m.mean_activity) AS activity,
  COALESCE(j.got, m.mean_got) AS got,
  COALESCE(j.gpt, m.mean_gpt) AS gpt,
  COALESCE(j.alb, m.mean_alb) AS albumin_level,
  COALESCE(j.tbil, m.mean_tbil) AS tbil,
  COALESCE(j.dbil, m.mean_dbil) AS dbil,
  COALESCE(j.che, m.mean_che) AS che,
  COALESCE(j.ttt, m.mean_ttt) AS ttt,
  COALESCE(j.ztt, m.mean_ztt) AS ztt,
  COALESCE(j.tcho, m.mean_tcho) AS tcho,
  COALESCE(j.tp, m.mean_tp) AS tp,
  
  -- Convert Type to a standardized target variable
  CASE 
    WHEN j.Type = 'AH' THEN 'Acute_Hepatitis'
    WHEN j.Type = 'CH' THEN 'Chronic_Hepatitis'
    WHEN j.Type = 'LC' THEN 'Liver_Cirrhosis'
    ELSE j.Type
  END AS target
FROM
  joined_data j,
  numerical_means m
WHERE
  j.row_num = 1  -- Remove duplicates by keeping only the first occurrence
ORDER BY 
  m_id;

-- Create additional view for normalized data (for ML tasks)
CREATE OR REPLACE VIEW `aise3010finalproject.hepatits.ml_ready_data` AS
WITH base_data AS (
  SELECT
    *,
    AVG(got) OVER() AS avg_got,
    STDDEV(got) OVER() AS std_got,
    AVG(gpt) OVER() AS avg_gpt,
    STDDEV(gpt) OVER() AS std_gpt,
    AVG(albumin_level) OVER() AS avg_alb,
    STDDEV(albumin_level) OVER() AS std_alb,
    AVG(tbil) OVER() AS avg_tbil,
    STDDEV(tbil) OVER() AS std_tbil,
    AVG(dbil) OVER() AS avg_dbil,
    STDDEV(dbil) OVER() AS std_dbil
  FROM
    `aise3010finalproject.hepatits.clean_hepatitis_data`
)
SELECT
  m_id,
  sex_standardized,
  age,
  fibros,
  activity,
  -- Safely normalize with NULLIF to prevent division by zero
  (got - avg_got) / NULLIF(std_got, 0) AS got_normalized,
  (gpt - avg_gpt) / NULLIF(std_gpt, 0) AS gpt_normalized,
  (albumin_level - avg_alb) / NULLIF(std_alb, 0) AS albumin_normalized,
  (tbil - avg_tbil) / NULLIF(std_tbil, 0) AS tbil_normalized,
  (dbil - avg_dbil) / NULLIF(std_dbil, 0) AS dbil_normalized,
  che,
  ttt,
  ztt,
  tcho,
  tp,
  target
FROM
  base_data;