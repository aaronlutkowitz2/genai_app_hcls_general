/************************
Aaron Wilkowitz 
Dicom Data
Create metadata on models
************************/

/************************
Create table with 
- image_id
- image_file_path
- facts_about_image
************************/

# Generate information tables
CREATE OR REPLACE TABLE `cloudadopt.dicom_mammography.image_model_input` as 
with meta_data as (
  SELECT SeriesInstanceUID, StudyInstanceUID
    # Series: 1.3.6.1.4.1.9590.100.1.2.370020768811676220424047464123530592133
    # Study: 1.3.6.1.4.1.9590.100.1.2.304153155112070673724116883611079168062
  FROM `cloudadopt.dicom_mammography.meta` 
  WHERE seriesdescription = 'full mammogram images' 
) 
, dicom_info_data as (
SELECT 
    a.SOPInstanceUID
  , a.SeriesInstanceUID
  , a.StudyInstanceUID
  , a.file_path 
  , a.image_path
  , a.BodyPartExamined
  , a.PatientOrientation
  , a.PatientID
  , a.PatientName
FROM `cloudadopt.dicom_mammography.dicom_info` a 
INNER JOIN meta_data b 
  ON a.SeriesInstanceUID = b.SeriesInstanceUID
  AND a.StudyInstanceUID = b.StudyInstanceUID
)
, calc_union_all as (
          SELECT 'test' as test_train, * FROM `cloudadopt.dicom_mammography.calc_case_description_test_set` 
UNION ALL SELECT 'train' as test_train, * FROM `cloudadopt.dicom_mammography.calc_case_description_train_set` 
)
, calc_join_back as (
SELECT 
    a.SeriesInstanceUID
  , a.StudyInstanceUID
  , a.image_path
  , a.BodyPartExamined
  , a.PatientOrientation
  , a.patientid 
  , b.test_train 
  , b.patient_id 
  , b.breast_density
  , b.left_or_right_breast
  , b.abnormality_id as calc_abnormality_id 
  , b.calc_type 
  , b.calc_distribution 
  , b.assessment as calc_assessment 
  , b.pathology as calc_pathology
  , b.subtlety as calc_subtlety
FROM dicom_info_data a 
INNER JOIN calc_union_all b 
  ON a.patientid = SPLIT(b.image_file_path, '/')[OFFSET(0)]
  AND a.StudyInstanceUID = SPLIT(b.image_file_path, '/')[OFFSET(1)]
  AND a.SeriesInstanceUID = SPLIT(b.image_file_path, '/')[OFFSET(2)]
)
, mass_union_all as (
          SELECT 'test' as test_train, * FROM `cloudadopt.dicom_mammography.mass_case_description_test_set` 
UNION ALL SELECT 'train' as test_train, * FROM `cloudadopt.dicom_mammography.mass_case_description_train_set` 
)
, mass_join_back as (
SELECT 
    a.SeriesInstanceUID
  , a.StudyInstanceUID
  , a.image_path
  , a.BodyPartExamined
  , a.PatientOrientation
  , a.patientid 
  , b.test_train 
  , b.patient_id 
  , b.breast_density
  , b.left_or_right_breast
  , b.abnormality_id as mass_abnormality_id 
  , b.mass_shape 
  , b.mass_margins 
  , b.assessment as mass_assessment 
  , b.pathology as mass_pathology
  , b.subtlety as mass_subtlety
FROM dicom_info_data a 
INNER JOIN mass_union_all b 
  ON a.patientid = SPLIT(b.image_file_path, '/')[OFFSET(0)]
  AND a.StudyInstanceUID = SPLIT(b.image_file_path, '/')[OFFSET(1)]
  AND a.SeriesInstanceUID = SPLIT(b.image_file_path, '/')[OFFSET(2)]
)
, combine_mass_calc as (
SELECT 
    coalesce(a.image_path, b.image_path) as image_path
  , coalesce(a.BodyPartExamined, b.BodyPartExamined) as BodyPartExamined
  , coalesce(a.PatientOrientation, b.PatientOrientation) as PatientOrientation
  , coalesce(a.breast_density, b.breast_density) as breast_density
  , coalesce(a.left_or_right_breast, b.left_or_right_breast) as left_or_right_breast
  , coalesce(a.test_train, b.test_train) as test_train
  , case when a.calc_type is null then 'mass' else 'calc' end as mass_vs_calc
  , a.calc_abnormality_id 
  , a.calc_type 
  , a.calc_distribution 
  , a.calc_assessment 
  , a.calc_pathology
  , a.calc_subtlety
  , b.mass_abnormality_id 
  , b.mass_shape 
  , b.mass_margins 
  , b.mass_assessment 
  , b.mass_pathology
  , b.mass_subtlety
FROM calc_join_back a 
FULL OUTER JOIN mass_join_back b 
  ON a.StudyInstanceUID = b.StudyInstanceUID
  AND a.SeriesInstanceUID = b.SeriesInstanceUID
  AND a.patientid = b.patientid
)
, update_image_path as (
SELECT 
    SPLIT(image_path, '/')[OFFSET(1)] as offset1 
  , SPLIT(image_path, '/')[OFFSET(2)] as offset2
  , SUBSTR(SPLIT(image_path, '/')[OFFSET(2)], length('1.3.6.1.4.1.9590.100.1.2.')+1,1) as file_name_str  
  , SPLIT(image_path, '/')[OFFSET(3)] as offset3 
  , * except(image_path)
FROM combine_mass_calc
)
, final as (
SELECT 
    'hcls/dicom/' || offset1 || '/file' || file_name_str || '/' || offset2 || '/' || offset3 as image_path 
  , * except(offset1, offset2, file_name_str, offset3)
FROM update_image_path
WHERE cast(file_name_str as float64) >= 4 
)
SELECT * 
FROM final ; 
# WHERE test_train = 'test'
# AND mass_vs_calc = 'calc'
# test_train, mass_vs_calc, count(*) 
# GROUP BY 1,2 
# GROUP BY 1 

CREATE OR REPLACE TABLE `cloudadopt.dicom_mammography.image_model_input_final` AS 
with max_only as (
  SELECT 
      image_path
    , max(load_time) as max_load_time 
  FROM `cloudadopt.dicom_mammography.image_model_embedding`
  GROUP BY 1
)
, combine_max as (
  SELECT a.* 
  FROM `cloudadopt.dicom_mammography.image_model_embedding` a 
  INNER JOIN max_only b 
    ON a.image_path = b.image_path 
    AND a.load_time = b.max_load_time
)
, create_pk as (
  SELECT 
    row_number() over (order by image_path) as id
    , * 
  FROM `cloudadopt.dicom_mammography.image_model_input`
)
SELECT 
    b.embedding_str 
  , a.* 
  , '{"input_text": "Context: You are an expert in reading medical images. Here is a text embedding of a mammogram image. Read it and provide information on the body part examined, patient orientation, breast density, left_or_right_breast, mass abnormality id, mass shape, mass margins, mass assessment, mass pathology, and mass subtlety. In the response, we expect a json output with a very specific pattern. embedding: ' || embedding_str || '",'
    || '"output_text": "{'
    || '"body_part_examined":"' || BodyPartExamined || '"'
    || ',"patient_orientation":"' || PatientOrientation || '"'
    || ',"breast_density":"' || breast_density || '"'
    || ',"left_or_right_breast":"' || left_or_right_breast || '"'
    || ',"mass_abnormality_id":"' || mass_abnormality_id || '"'
    || ',"mass_shape":"' || mass_shape || '"'
    || ',"mass_margins":"' || mass_margins || '"'
    || ',"mass_assessment":"' || mass_assessment || '"'
    || ',"mass_pathology":"' || mass_pathology || '"'
    || ',"mass_subtlety":"' || mass_subtlety || '"'
    || '}"}' as jsonl_full
  # , '{"id": ' || id || ', "embedding": [' || embedding_str || ']}' as jsonl_index
FROM create_pk a 
INNER JOIN combine_max b
  ON a.image_path = b.image_path 
; 

/************************
Calc Nearest Neighbors
************************/

CREATE OR REPLACE TABLE `cloudadopt.dicom_mammography.image_model_input_1` AS 
WITH pretable as (
  SELECT 
    id
  , replace(replace(embedding_str,'[',''),']','') as embedding_str
  , test_train
  , mass_vs_calc
  FROM `cloudadopt.dicom_mammography.image_model_input_final`
)
SELECT 
    id
  , ARRAY(SELECT CAST(num AS FLOAT64) FROM UNNEST(SPLIT(embedding_str)) num) as embedding
  , test_train
  , mass_vs_calc
FROM pretable 
;

CREATE OR REPLACE TABLE `cloudadopt.dicom_mammography.image_model_input_2` AS 
SELECT 
    a.id 
  , a.embedding 
  , b.id as id_neighbor
  , b.embedding as embedding_neighbor
FROM `cloudadopt.dicom_mammography.image_model_input_1` a 
LEFT JOIN `cloudadopt.dicom_mammography.image_model_input_1` b
  ON a.mass_vs_calc = b.mass_vs_calc
  AND a.id <> b.id
;

# 254,282
CREATE OR REPLACE TABLE `cloudadopt.dicom_mammography.image_model_input_3` AS 
SELECT *,
  (SELECT SUM(element1 * element2) 
    FROM t.embedding element1 WITH OFFSET pos
    JOIN t.embedding_neighbor element2 WITH OFFSET pos 
    USING(pos)
  ) dot_product
FROM `cloudadopt.dicom_mammography.image_model_input_2` t
; 

CREATE OR REPLACE TABLE `cloudadopt.dicom_mammography.image_model_input_4` AS 
with predata as (
SELECT 
    a.id
  , a.id_neighbor
  , dot_product
  , pow(dot_product,10) as dot_product10
  , pow(dot_product,100) as dot_product100
  , b.bodypartexamined
  , c.bodypartexamined as bodypartexamined_neighbor
  , b.patientorientation
  , c.patientorientation as patientorientation_neighbor
  , b.breast_density
  , c.breast_density as breast_density_neighbor
  , b.left_or_right_breast
  , c.left_or_right_breast as left_or_right_breast_neighbor
  , b.mass_abnormality_id
  , c.mass_abnormality_id as mass_abnormality_id_neighbor
  , b.mass_shape
  , c.mass_shape as mass_shape_neighbor
  , b.mass_margins
  , c.mass_margins as mass_margins_neighbor
  , b.mass_assessment
  , c.mass_assessment as mass_assessment_neighbor
  , b.mass_pathology
  , c.mass_pathology as mass_pathology_neighbor
  , b.mass_subtlety
  , c.mass_subtlety as mass_subtlety_neighbor
FROM `cloudadopt.dicom_mammography.image_model_input_3` a 
LEFT JOIN `cloudadopt.dicom_mammography.image_model_input_final` b 
  ON a.id = b.id 
LEFT JOIN `cloudadopt.dicom_mammography.image_model_input_final` c
  ON a.id_neighbor = c.id 
WHERE c.test_train = 'test' # only test included in neighbors exercise
AND b.mass_vs_calc = 'mass'
AND c.mass_vs_calc = 'mass'
ORDER by 1,3 desc 
)
SELECT 
    id
  , id_neighbor
  , rank() over (partition by id order by dot_product desc) as rank_neighbor
  , * except (id, id_neighbor)
FROM predata
ORDER BY 1,3
;

CREATE OR REPLACE TABLE `cloudadopt.dicom_mammography.image_model_input_5` AS 
          SELECT id, id_neighbor, rank_neighbor, dot_product, dot_product10, dot_product100, 'body_part_examined' as measurement, bodypartexamined as value, bodypartexamined_neighbor as value_neighbor FROM `cloudadopt.dicom_mammography.image_model_input_4`
UNION ALL SELECT id, id_neighbor, rank_neighbor, dot_product, dot_product10, dot_product100, 'patient_orientation' as measurement, patientorientation as value, patientorientation_neighbor as value_neighbor FROM `cloudadopt.dicom_mammography.image_model_input_4`
UNION ALL SELECT id, id_neighbor, rank_neighbor, dot_product, dot_product10, dot_product100, 'breast_density' as measurement, cast(breast_density as string) as value, cast(breast_density_neighbor as string) as value_neighbor FROM `cloudadopt.dicom_mammography.image_model_input_4`
UNION ALL SELECT id, id_neighbor, rank_neighbor, dot_product, dot_product10, dot_product100, 'left_right_breast' as measurement, left_or_right_breast as value, left_or_right_breast_neighbor as value_neighbor FROM `cloudadopt.dicom_mammography.image_model_input_4`
UNION ALL SELECT id, id_neighbor, rank_neighbor, dot_product, dot_product10, dot_product100, 'mass_abnormality_num' as measurement, cast(mass_abnormality_id as string) as value, cast(mass_abnormality_id_neighbor as string) as value_neighbor FROM `cloudadopt.dicom_mammography.image_model_input_4`
UNION ALL SELECT id, id_neighbor, rank_neighbor, dot_product, dot_product10, dot_product100, 'mass_shape' as measurement, mass_shape as value, mass_shape_neighbor as value_neighbor FROM `cloudadopt.dicom_mammography.image_model_input_4`
UNION ALL SELECT id, id_neighbor, rank_neighbor, dot_product, dot_product10, dot_product100, 'mass_margins' as measurement, mass_margins as value, mass_margins_neighbor as value_neighbor FROM `cloudadopt.dicom_mammography.image_model_input_4`
UNION ALL SELECT id, id_neighbor, rank_neighbor, dot_product, dot_product10, dot_product100, 'mass_assessment' as measurement, cast(mass_assessment as string) as value, cast(mass_assessment_neighbor as string) as value_neighbor FROM `cloudadopt.dicom_mammography.image_model_input_4`
UNION ALL SELECT id, id_neighbor, rank_neighbor, dot_product, dot_product10, dot_product100, 'mass_pathology' as measurement, mass_pathology as value, mass_pathology_neighbor as value_neighbor FROM `cloudadopt.dicom_mammography.image_model_input_4`
UNION ALL SELECT id, id_neighbor, rank_neighbor, dot_product, dot_product10, dot_product100, 'mass_subtlety' as measurement, cast(mass_subtlety as string) as value, cast(mass_subtlety_neighbor as string) as value_neighbor FROM `cloudadopt.dicom_mammography.image_model_input_4`
;

-- ## Mass Train
-- SELECT a.* 
-- FROM `cloudadopt.dicom_mammography.image_model_input_final_ann_index_input` a
-- INNER JOIN `cloudadopt.dicom_mammography.image_model_input_final` b 
--   ON a.id = b.id 
-- WHERE test_train = 'train' 
-- AND mass_vs_calc = 'mass' 
-- ; 

-- ## Calc Train
-- SELECT a.* 
-- FROM `cloudadopt.dicom_mammography.image_model_input_final_ann_index_input` a
-- INNER JOIN `cloudadopt.dicom_mammography.image_model_input_final` b 
--   ON a.id = b.id 
-- WHERE test_train = 'train' 
-- AND mass_vs_calc = 'calc'
-- ;
 

/************************
Prework
************************/

-- SELECT modality, seriesdescription, BodyPartExamined, SeriesNumber, collection, sum(ImageCount)
-- FROM `cloudadopt.dicom_mammography.meta` 
-- GROUP BY 1,2,3,4,5
-- ORDER BY 6 desc ; 

-- # Counts by series 
--   # 7026 ROI Mask images 
--   # 3103 Full Mammogram images 
--   # 110 Cropped images
-- SELECT seriesdescription, sum(ImageCount)
-- FROM `cloudadopt.dicom_mammography.meta` 
-- GROUP BY 1
-- ORDER BY 2 desc ; 

-- # Confirm series, study ID is PK for dicom data -- it is for "full mammogram images", not for other data types --> Let's focus on just full mammogram images
-- with meta_data as (
--   SELECT SeriesInstanceUID, StudyInstanceUID
--     # Series: 1.3.6.1.4.1.9590.100.1.2.370020768811676220424047464123530592133
--     # Study: 1.3.6.1.4.1.9590.100.1.2.304153155112070673724116883611079168062
--   FROM `cloudadopt.dicom_mammography.meta` 
--   WHERE seriesdescription = 'full mammogram images' 
-- ) 
-- , pre_data as (
--   SELECT 
--       a.SOPInstanceUID
--     , a.SeriesInstanceUID
--     , a.StudyInstanceUID
--     , a.file_path 
--     , a.image_path
--     , a.BodyPartExamined
--     , a.PatientOrientation
--     , a.PatientID
--     , a.PatientName
--   FROM `cloudadopt.dicom_mammography.dicom_info` a 
--   JOIN meta_data b 
--     ON a.SeriesInstanceUID = b.SeriesInstanceUID
--     AND a.StudyInstanceUID = b.StudyInstanceUID
--   GROUP BY 1,2,3,4,5,6,7,8,9
-- )
-- SELECT 
--     a.SeriesInstanceUID
--   , a.StudyInstanceUID
--   , count(*)
-- FROM pre_data a 
-- GROUP BY 1,2
-- HAVING count(*) > 1 
-- ORDER BY 3 desc ; 



/************************
Graveyard
************************/


-- with calc_union_all as (
--           SELECT 'test' as test_train, * FROM `cloudadopt.dicom_mammography.calc_case_description_test_set` 
-- UNION ALL SELECT 'train' as test_train, * FROM `cloudadopt.dicom_mammography.calc_case_description_train_set` 
-- )
-- , mass_union_all as (
--           SELECT 'test' as test_train, * FROM `cloudadopt.dicom_mammography.mass_case_description_test_set` 
-- UNION ALL SELECT 'train' as test_train, * FROM `cloudadopt.dicom_mammography.mass_case_description_train_set` 
-- )
-- SELECT * -- a.patient_id, count(*) 
-- FROM calc_union_all a
-- JOIN mass_union_all b 
--   ON a.patient_id = b.patient_id
-- WHERE a.patient_id = 'P_00106'
-- -- GROUP BY 1 
-- -- ORDER BY 2 desc 

-- SELECT * 
-- FROM `cloudadopt.dicom_mammography.calc_case_description_test_set` 
-- WHERE SPLIT(image_file_path, '/')[OFFSET(2)] = '1.3.6.1.4.1.9590.100.1.2.370020768811676220424047464123530592133'

-- SELECT * 
-- FROM `cloudadopt.dicom_mammography.calc_case_description_train_set` 
-- WHERE SPLIT(image_file_path, '/')[OFFSET(1)] = '1.3.6.1.4.1.9590.100.1.2.370020768811676220424047464123530592133'

-- SELECT * 
-- FROM `cloudadopt.dicom_mammography.mass_case_description_test_set` 
-- WHERE SPLIT(image_file_path, '/')[OFFSET(1)] = '1.3.6.1.4.1.9590.100.1.2.370020768811676220424047464123530592133'

-- SELECT * 
-- FROM `cloudadopt.dicom_mammography.mass_case_description_train_set` 
-- WHERE SPLIT(image_file_path, '/')[OFFSET(2)] = '1.3.6.1.4.1.9590.100.1.2.370020768811676220424047464123530592133'
-- AND SPLIT(image_file_path, '/')[OFFSET(1)] = '1.3.6.1.4.1.9590.100.1.2.304153155112070673724116883611079168062'

-- SELECT 
--     image_file_path
--   , SPLIT(image_file_path, '/')[OFFSET(0)] as p0
--   , SPLIT(image_file_path, '/')[OFFSET(1)] as p1
--   , SPLIT(image_file_path, '/')[OFFSET(2)] as p2
--   , SPLIT(image_file_path, '/')[OFFSET(3)] as p3 
--   -- , SPLIT(image_file_path, '/')[OFFSET(4)]
--   -- , SPLIT(image_file_path, '/')[OFFSET(5)]
-- FROM `cloudadopt.dicom_mammography.calc_case_description_test_set` 
-- LIMIT 100 

-- # Metadata 
--   # CBIS-DDSM/dicom/1.3.6.1.4.1.9590.100.1.2.370020768811676220424047464123530592133/1-204.dcm
--   # CBIS-DDSM/jpeg/1.3.6.1.4.1.9590.100.1.2.370020768811676220424047464123530592133/1-204.jpg


-- SELECT * 
-- FROM `cloudadopt.dicom_mammography.mass_case_description_test_set` b
-- WHERE SPLIT(b.image_file_path, '/')[OFFSET(2)] = '1.3.6.1.4.1.9590.100.1.2.361153176312829994122249772871569233533' 
-- AND SPLIT(b.image_file_path, '/')[OFFSET(1)] = '1.3.6.1.4.1.9590.100.1.2.165414265713200150806512436912467246584'
-- AND SPLIT(b.image_file_path, '/')[OFFSET(0)] = 'Calc-Test_P_01534_LEFT_CC_1'


-- with meta_data as (
--   SELECT SeriesInstanceUID, StudyInstanceUID
--     # Series: 1.3.6.1.4.1.9590.100.1.2.370020768811676220424047464123530592133
--     # Study: 1.3.6.1.4.1.9590.100.1.2.304153155112070673724116883611079168062
--   FROM `cloudadopt.dicom_mammography.meta` 
--   WHERE seriesdescription = 'full mammogram images' 
-- ) 
-- , pre_data as (
--   SELECT 
--       a.SOPInstanceUID
--     , a.SeriesInstanceUID
--     , a.StudyInstanceUID
--     , a.file_path 
--     , a.image_path
--     , a.BodyPartExamined
--     , a.PatientOrientation
--     , a.PatientID
--     , a.PatientName
--   FROM `cloudadopt.dicom_mammography.dicom_info` a 
--   JOIN meta_data b 
--     ON a.SeriesInstanceUID = b.SeriesInstanceUID
--     AND a.StudyInstanceUID = b.StudyInstanceUID
--   GROUP BY 1,2,3,4,5,6,7,8,9
-- )
-- # , pre_data2 as (
-- SELECT 
--     a.SeriesInstanceUID
--   , a.StudyInstanceUID
--   , a.patientID
--   , count(*) as count 
-- FROM pre_data a 
-- GROUP BY 1,2,3
-- # HAVING count(*) > 1
-- ORDER BY 4 desc 
-- ) 
-- SELECT * 
-- FROM `cloudadopt.dicom_mammography.mass_case_description_train_set` a
-- JOIN pre_data2 b 
--   ON b.StudyInstanceUID = SPLIT(a.image_file_path, '/')[OFFSET(1)]
--   AND b.SeriesInstanceUID = SPLIT(a.image_file_path, '/')[OFFSET(2)]