
CREATE OR REPLACE VIEW `aise3010finalproject.hepatits.feature_engineering` AS
SELECT
  d.sex,
  d.age,
  b.fibros,
  b.activity,
  i.got,
  i.gpt,
  i.alb AS albumin_level,  -- Select alb once, renamed for clarity
  i.tbil,
  i.dbil,
  i.che,
  i.ttt,
  i.ztt,
  i.tcho,
  i.tp,
  d.Type AS target  -- Renamed for ML clarity
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
  d.Type IS NOT NULL;  -- Ensure target is not null (from step c))