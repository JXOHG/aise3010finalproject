-- Stratified training data
CREATE OR REPLACE TABLE `aise3010finalproject.hepatits.train_data` AS
WITH RankedData AS (
  SELECT
    *,
    ROW_NUMBER() OVER(PARTITION BY target ORDER BY RAND()) AS rn,
    COUNT(*) OVER(PARTITION BY target) AS class_count
  FROM
    `aise3010finalproject.hepatits.feature_engineering`
)
SELECT
  * EXCEPT(rn, class_count)
FROM
  RankedData
WHERE
  rn <= 0.8 * class_count;

-- Stratified testing data
CREATE OR REPLACE TABLE `aise3010finalproject.hepatits.test_data` AS
WITH RankedData AS (
  SELECT
    *,
    ROW_NUMBER() OVER(PARTITION BY target ORDER BY RAND()) AS rn,
    COUNT(*) OVER(PARTITION BY target) AS class_count
  FROM
    `aise3010finalproject.hepatits.feature_engineering`
)
SELECT
  * EXCEPT(rn, class_count)
FROM
  RankedData
WHERE
  rn > 0.8 * class_count;