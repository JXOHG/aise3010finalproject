EXPORT DATA
  OPTIONS (
    uri = 'gs://aise3010finalproject-bucket/train_data/*.csv',
    format = 'CSV',
    overwrite = TRUE,
    header = TRUE
  ) AS
SELECT
  *
FROM
  `aise3010finalproject.hepatits.train_data`;

EXPORT DATA
  OPTIONS (
    uri = 'gs://aise3010finalproject-bucket/test_data/*.csv',
    format = 'CSV',
    overwrite = TRUE,
    header = TRUE
  ) AS
SELECT
  *
FROM
  `aise3010finalproject.hepatits.test_data`;

EXPORT DATA
  OPTIONS (
    uri = 'gs://aise3010finalproject-bucket/feature_engineering/*.csv',
    format = 'CSV',
    overwrite = TRUE,
    header = TRUE
  ) AS
SELECT
  *
FROM
  `aise3010finalproject.hepatits.feature_engineering`;