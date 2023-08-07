/*******************
HRG Pipeline Health
Aaron Wilkowitz 
2023-03-28 
*******************/

/*******************
Section 1: Create Table 1 - Pipeline Health
*******************/

CREATE OR REPLACE TABLE `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_1_month_widen`  AS 
with entity_id_add as (
SELECT
    row_number() over (order by 'x') as entity_id
  , group_name
  , division_name
  , market_name
  , coid
  , coid_name
  , functional_dept_desc
  , sub_functional_dept_desc
  , skillmix
  , status
  , rn_experience
  , contract_type
  , year
  , month
  , type 
FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_data_20230331` 
GROUP BY 2,3,4,5,6,7,8,9,10,11,12,13,14,15
)
, entity_back_to_full_table as (
  SELECT 
      b.entity_id 
    , a.*
    , date(a.year,a.month,1) as measurement_date
  FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_data_20230331` a 
  JOIN entity_id_add b 
    ON  coalesce(a.group_name,'x') = coalesce(b.group_name,'x')
    AND coalesce(a.division_name,'x') = coalesce(b.division_name,'x')
    AND coalesce(a.market_name,'x') = coalesce(b.market_name,'x')
    AND coalesce(a.coid,-99) = coalesce(b.coid,-99)
    AND coalesce(a.coid_name,'x') = coalesce(b.coid_name,'x')
    AND coalesce(a.functional_dept_desc,'x') = coalesce(b.functional_dept_desc,'x')
    AND coalesce(a.sub_functional_dept_desc,'x') = coalesce(b.sub_functional_dept_desc,'x')
    AND coalesce(a.skillmix,-99) = coalesce(b.skillmix,-99)
    AND coalesce(a.status,'x') = coalesce(b.status,'x')
    AND coalesce(a.rn_experience,'x') = coalesce(b.rn_experience,'x')
    AND coalesce(a.contract_type,'x') = coalesce(b.contract_type,'x')
    AND coalesce(a.year,-99) = coalesce(b.year,-99)
    AND coalesce(a.month,-99) = coalesce(b.month,-99)
    AND coalesce(a.type,'x') = coalesce(b.type,'x')
)
, pivot_pre as (
  SELECT 
      entity_id
    , replace(replace(lower(metric_name),' ','_'),'-','_') as metric_name
    , type 
    , rn_experience 
    , contract_type 
    , case when metric_name = 'Terms' then counter * -1 else counter end as counter 
  FROM entity_back_to_full_table
  GROUP BY 1,2,3,4,5,6
)
, pivot_function as (
  SELECT * 
  FROM pivot_pre
  PIVOT
  (
    -- #2 aggregate
    SUM(counter) AS total
    -- #3 pivot_column
    FOR metric_name in (
        'headcount'
      , 'hires'
      , 'terms'
      , 'skillmix_in'
      , 'skillmix_out'
      , 'status_change_in'
      , 'status_change_out'
      , 'vacancies'
      , 'non_distinct_headcount'
    )
  )
)
, coid_info_distinct as (
  SELECT 
      coid
    , facility_name
    , group_name
    , division_name
    , market_name
    , lat
    , long
  FROM `hca-sandbox-aaron-argolis.irc.bm_pre_combined_table`
  GROUP BY 1,2,3,4,5,6,7
)
, add_coid_info as (
  SELECT 
      a.* except(group_name, division_name, market_name, coid_name)
    , b.facility_name
    , b.group_name
    , b.division_name
    , b.market_name
    , b.lat
    , b.long
  FROM entity_back_to_full_table a 
  LEFT JOIN coid_info_distinct b 
    ON a.coid = b.coid
)
, final_table as (
SELECT 
    row_number() over (order by a.entity_id) as row_id
  , a.* except(metric_name, type, rn_experience, contract_type, counter)
  , b.* except(entity_id)
  , b.total_vacancies + b.total_non_distinct_headcount as total_target_headcount
FROM add_coid_info a 
JOIN pivot_function b 
  ON a.entity_id = b.entity_id
)
SELECT *
FROM final_table
;

/*******************
Section 2: Create Table 2 - Recruiting Health
*******************/

CREATE OR REPLACE TABLE `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_recruiting_pipeline_sample_data_widen`  AS 
with coid_info_distinct as (
  SELECT 
      coid
    , facility_name
    , group_name
    , division_name
    , market_name
    , lat
    , long
  FROM `hca-sandbox-aaron-argolis.irc.bm_pre_combined_table`
  GROUP BY 1,2,3,4,5,6,7
)
, add_coid_info as (
  SELECT 
      a.* except(group_name, division_name, market_name, coid_name, emp_status, screen_date, interview_date)
    , b.facility_name
    , b.group_name
    , b.division_name
    , b.market_name
    , b.lat
    , b.long
    , a.emp_status 
    , cast(screen_date as date) as screen_date
    , case 
        when interview_date is null and extend_offer_date is not null then date_add(cast(screen_date as date), interval 1 day) # if they've already been interviewed, just set the interview date as 1 day after screen date 
        else cast(interview_date as date) 
      end as interview_date 
  FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_recruiting_pipeline_sample_data_20230331` a 
  LEFT JOIN coid_info_distinct b 
    ON a.coid = b.coid
)
, create_binary as (
  SELECT 
      * 
    , case when screen_date is not null then TRUE else FALSE end app_screen
    , case when interview_date is not null then TRUE else FALSE end screen_int
    , case when extend_offer_date is not null then TRUE else FALSE end int_offer
    , case when accept_offer_date is not null then TRUE else FALSE end offer_accept
    , case when offer_start_date is not null then TRUE else FALSE end accept_hire
  FROM add_coid_info
)
SELECT *
FROM create_binary
; 


### Create a snapshot by day

CREATE OR REPLACE TABLE `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_recruiting_pipeline_sample_data_widen_daily_snapshot`  AS 
with rows10 as (
            SELECT 1 as counter_id
  UNION ALL SELECT 2
  UNION ALL SELECT 3
  UNION ALL SELECT 4
  UNION ALL SELECT 5
  UNION ALL SELECT 6
  UNION ALL SELECT 7
  UNION ALL SELECT 8
  UNION ALL SELECT 9
  UNION ALL SELECT 10
)
, rows365 as (
  SELECT row_number() over (order by 'x') as days_since_app 
  FROM rows10 a -- 10
  , rows10 b -- 100 
  , rows10 c -- 1k
  LIMIT 365
)
, every_stage_by_row as (
  SELECT 
      application_id  
    , date_diff(application_date, application_date, day) as days_to_app
    , date_diff(screen_date, application_date, day) as days_to_screen
    , date_diff(interview_date, application_date, day) as days_to_int
    , date_diff(extend_offer_date, application_date, day) as days_to_offer
    , date_diff(accept_offer_date, application_date, day) as days_to_accept
    , date_diff(offer_start_date, application_date, day) as days_to_hire
  FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_recruiting_pipeline_sample_data_widen`
)
, max_until_out as (
  SELECT 
      *
    , case 
        when days_to_screen is null then days_to_app
        when days_to_int is null then days_to_screen
        when days_to_offer is null then days_to_int
        when days_to_accept is null then days_to_offer
        when days_to_hire is null then days_to_accept
        else days_to_hire
      end as max_days_until_out 
    , case 
        when days_to_screen is null then 'F Application Complete'
        when days_to_int is null then 'E Screen Complete'
        when days_to_offer is null then 'D Interview Complete'
        when days_to_accept is null then 'C Offer Sent'
        when days_to_hire is null then 'B Offer Accepted'
        else 'A Started'
      end as max_status_final
  FROM every_stage_by_row
)
, combine_365_with_every_stage as (
  SELECT 
      b.application_id 
    , a.days_since_app
    , floor(a.days_since_app / (365/52)) as weeks_since_app
    , floor(a.days_since_app / (365/12)) as months_since_app
    , case 
        when a.days_since_app > b.max_days_until_out then b.max_status_final
        when a.days_since_app >= b.days_to_app and a.days_since_app < coalesce(b.days_to_screen,999) then 'F Application Complete'
        when a.days_since_app >= b.days_to_screen and a.days_since_app < coalesce(b.days_to_int,999) then 'E Screen Complete'
        when a.days_since_app >= b.days_to_int and a.days_since_app < coalesce(b.days_to_offer,999) then 'D Interview Complete'
        when a.days_since_app >= b.days_to_offer and a.days_since_app < coalesce(b.days_to_accept,999) then 'C Offer Sent'
        when a.days_since_app >= b.days_to_accept and a.days_since_app < coalesce(b.days_to_hire,999) then 'B Offer Accepted'
        when a.days_since_app >= b.days_to_hire then 'A Started'
        else 'Unknown'
      end as status
  FROM rows365 a 
  , max_until_out b
)
SELECT  
    a.application_id 
  , a.days_since_app
  , min(status) as status
# , b.*
FROM combine_365_with_every_stage a 
-- LEFT JOIN `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_recruiting_pipeline_sample_data_widen` b
--   ON a.application_id = b.application_id 
# FROM max_until_out
# WHERE days_to_hire is not null 
-- WHERE offer_start_date is not null 
GROUP BY 1,2
ORDER BY 1,2
;

/*******************
Section 3: Create Table 3 - Combine Datsets
*******************/

## Pipeline health
CREATE OR REPLACE TABLE `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_what_if_scenario`  AS 
with coid_func_sub_func_combine as (
    SELECT 
        coid
      , functional_dept_desc
      , sub_functional_dept_desc
    FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_1_month_widen`
    GROUP BY 1,2,3
  UNION ALL 
    SELECT 
        coid
      , functional_dept_desc
      , sub_functional_dept_desc
    FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_recruiting_pipeline_sample_data_widen`
    GROUP BY 1,2,3
)
, coid_func_sub_func_combine_distinct as (
  SELECT *
  FROM coid_func_sub_func_combine
  GROUP BY 1,2,3
)
, coid_func_sub_func_combine_distinct_id as (
  SELECT 
      row_number() over (order by 'x') as coid_func_sub_func_id
    , *
  FROM coid_func_sub_func_combine_distinct
  GROUP BY 2,3,4
)
, add_id_pipeline as (
  SELECT 
      b.coid_func_sub_func_id
    , a.*
  FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_1_month_widen` a 
  JOIN coid_func_sub_func_combine_distinct_id b 
    ON a.coid = b.coid
    AND a.functional_dept_desc = b.functional_dept_desc
    AND a.sub_functional_dept_desc = b.sub_functional_dept_desc
)
, sum_months_pipeline as (
  SELECT 
      coid_func_sub_func_id
    , count(distinct(measurement_date)) as count_distinct_months
  FROM add_id_pipeline
  GROUP BY 1 
)
, transfers_by_month as (
  SELECT 
      a.coid_func_sub_func_id
    , sum(a.total_hires) / max(b.count_distinct_months) as avg_hires	
    , sum(a.total_terms) / max(b.count_distinct_months) as avg_terms	
    , sum(a.total_skillmix_in) / max(b.count_distinct_months) as avg_skillmix_in
    , sum(a.total_skillmix_out) / max(b.count_distinct_months) as avg_skillmix_out
    , sum(a.total_status_change_in) / max(b.count_distinct_months) as avg_status_change_in
    , sum(a.total_status_change_out	) / max(b.count_distinct_months) as avg_status_change_out	
  FROM add_id_pipeline a 
  JOIN sum_months_pipeline b
    ON a.coid_func_sub_func_id = b.coid_func_sub_func_id
  GROUP BY 1 
)
, max_month_pipeline as (
  SELECT 
      coid_func_sub_func_id
    , max(measurement_date) as max_measurement_date
  FROM add_id_pipeline
  GROUP BY 1 
)
, headcount_at_max as (
  SELECT 
      a.coid_func_sub_func_id
    , sum(a.total_headcount) as total_headcount
    , sum(a.total_target_headcount) as total_target_headcount
  FROM add_id_pipeline a 
  JOIN max_month_pipeline b
    ON a.coid_func_sub_func_id = b.coid_func_sub_func_id
    AND a.measurement_date = b.max_measurement_date
  GROUP BY 1 
)
, combine_pipeline as (
SELECT 
    a.coid_func_sub_func_id
  , a.group_name
  , a.division_name
  , a.market_name
  , a.coid
  , a.facility_name as coid_name 
  , a.functional_dept_desc
  , a.sub_functional_dept_desc
  , avg(b.avg_hires) as avg_hires
  , avg(b.avg_terms) as avg_terms
  , avg(b.avg_skillmix_in) as avg_skillmix_in
  , avg(b.avg_skillmix_out) as avg_skillmix_out
  , avg(b.avg_status_change_in) as avg_status_change_in
  , avg(b.avg_status_change_out) as avg_status_change_out
  , avg(c.total_headcount) as total_headcount
  , avg(c.total_target_headcount) as total_target_headcount
FROM add_id_pipeline a 
LEFT JOIN transfers_by_month b 
  ON a.coid_func_sub_func_id = b.coid_func_sub_func_id
LEFT JOIN headcount_at_max c 
  ON a.coid_func_sub_func_id = c.coid_func_sub_func_id
GROUP BY 1,2,3,4,5,6,7,8
)
, add_id_recruting as (
  SELECT 
      b.coid_func_sub_func_id
    , a.*
    , date_diff(offer_start_date, application_date, day) as days_application_to_hire
  FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_recruiting_pipeline_sample_data_widen` a 
  JOIN  coid_func_sub_func_combine_distinct_id b 
    ON a.coid = b.coid
    AND a.functional_dept_desc = b.functional_dept_desc
    AND a.sub_functional_dept_desc = b.sub_functional_dept_desc
)
, sum_months_recruiting as (
  SELECT 
      coid_func_sub_func_id
    , count(distinct(application_date)) as count_distinct_months
  FROM add_id_recruting
  GROUP BY 1 
)
, applications_by_month as (
  SELECT 
      a.coid_func_sub_func_id
    , count(*) / max(b.count_distinct_months) as avg_applicants	
  FROM add_id_recruting a
  JOIN sum_months_recruiting b
    ON a.coid_func_sub_func_id = b.coid_func_sub_func_id
  GROUP BY 1 
)
, combine_recruiting as (
  SELECT 
      a.coid_func_sub_func_id
    , b.avg_applicants
    , COUNT(CASE WHEN a.app_screen THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_app_screen
    , COUNT(CASE WHEN a.screen_int THEN 1 ELSE NULL END) / nullif(COUNT(CASE WHEN a.app_screen THEN 1 ELSE NULL END), 0) AS perc_screen_int
    , COUNT(CASE WHEN a.int_offer THEN 1 ELSE NULL END) / nullif(COUNT(CASE WHEN a.screen_int THEN 1 ELSE NULL END), 0) AS perc_int_offer
    , COUNT(CASE WHEN a.offer_accept THEN 1 ELSE NULL END) / nullif(COUNT(CASE WHEN a.int_offer THEN 1 ELSE NULL END), 0) AS perc_offer_accept
    , COUNT(CASE WHEN a.accept_hire THEN 1 ELSE NULL END) / nullif(COUNT(CASE WHEN a.offer_accept THEN 1 ELSE NULL END), 0) AS perc_accept_hire
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (0 *  365/12) AND (1 *  365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_01_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (1 *  365/12) AND (2 *  365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_02_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (2 *  365/12) AND (3 *  365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_03_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (3 *  365/12) AND (4 *  365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_04_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (4 *  365/12) AND (5 *  365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_05_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (5 *  365/12) AND (6 *  365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_06_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (6 *  365/12) AND (7 *  365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_07_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (7 *  365/12) AND (8 *  365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_08_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (8 *  365/12) AND (9 *  365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_09_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (9 *  365/12) AND (10 * 365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_10_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (10 * 365/12) AND (11 * 365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_11_month
    , COUNT(CASE WHEN a.days_application_to_hire BETWEEN (11 * 365/12) AND (12 * 365/12) THEN 1 ELSE NULL END) / nullif(COUNT(*), 0) AS perc_hire_12_month
  FROM add_id_recruting a 
  LEFT JOIN applications_by_month b 
    ON a.coid_func_sub_func_id = b.coid_func_sub_func_id
  GROUP BY 1,2
)
SELECT
    a.*
  , b.* except(coid_func_sub_func_id)
FROM combine_pipeline a 
LEFT JOIN combine_recruiting b 
  ON a.coid_func_sub_func_id = b.coid_func_sub_func_id
; 




-- /*******************
-- Section 2 extra: Create Table 2 - Recruiting Health -- Fake Data
-- *******************/

-- ### Create the normal table

-- CREATE OR REPLACE TABLE `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_recruiting_pipeline_fake_data`  AS 
-- with rows10 as (
--             SELECT 1 as counter_id
--   UNION ALL SELECT 2
--   UNION ALL SELECT 3
--   UNION ALL SELECT 4
--   UNION ALL SELECT 5
--   UNION ALL SELECT 6
--   UNION ALL SELECT 7
--   UNION ALL SELECT 8
--   UNION ALL SELECT 9
--   UNION ALL SELECT 10
-- )
-- , rows3200 as (
--   SELECT row_number() over (order by 'x') as row_id 
--   FROM rows10 a -- 10
--   , rows10 b -- 100 
--   , rows10 c -- 1k
--   , rows10 d -- 10k
--   LIMIT 3200
-- )
-- , rows3200_into16 as (
--   SELECT 
--       row_id 
--     , mod(row_id,16) + 1 as coid_id
--   FROM rows3200 
-- )
-- , coid_list as (
--   SELECT 
--       row_number() over (order by coid) as coid_id 
--     , coid
--   FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_1_month_fake_data` 
--   GROUP BY 2
-- )
-- , combine_rows_to_coid as (
--   SELECT 
--       a.row_id
--     , b.coid
--     ### Create 3200 application dates -- 200/hosp
--     ### Create 1600 screen dates - - on aveage 45 days prior -- 100/hosp    
--     , round(rand() * 90,0) as days_app_screen
--     , rand() as prob_app_screen
--     ### Create 1200 interview date -- on average 30 days prior -- 75/hosp
--     , round(rand() * 60,0) as days_screen_int
--     , rand() as prob_screen_int
--     ### Create 800 review date -- on average 15 days prior -- 50/hosp
--     , round(rand() * 30,0) as days_int_review
--     , rand() as prob_int_review
--     ### Create 400 extend offer date - 250 accept, 150 reject -- on average 15 days prior -- 25/hosp
--     , round(rand() * 30,0) as days_review_offer
--     , rand() as prob_review_offer
--     ### Create 160 new hires dates from 2022-02-01 to 2023-01-01, 15 days prior -- 10/hosp
--     , round(rand() * 365,0) as days_after_start
--     , round(rand() * 10,0) as days_offer_hire
--     , rand() as prob_offer_hire
--   FROM rows3200_into16 a 
--   JOIN coid_list b 
--     ON a.coid_id = b.coid_id 
-- )
-- , create_outcomes as (
--   SELECT 
--       row_id 
--     , coid
--     , days_app_screen
--     , case when prob_app_screen < 0.75 then TRUE else FALSE END as app_screen 
--     , days_screen_int
--     , case when prob_app_screen < 0.75 and prob_screen_int < 0.4 then TRUE else FALSE END as screen_int
--     , days_int_review
--     , case when prob_app_screen < 0.75 and prob_screen_int < 0.4 and prob_int_review < 0.8 then TRUE else FALSE END as int_review
--     , days_review_offer
--     , case when prob_app_screen < 0.75 and prob_screen_int < 0.4 and prob_int_review < 0.8 and prob_review_offer < 0.4 then TRUE else FALSE END as review_offer
--     , days_offer_hire
--     , case when prob_app_screen < 0.75 and prob_screen_int < 0.4 and prob_int_review < 0.8 and prob_review_offer < 0.4 and prob_offer_hire < 0.6 then TRUE else FALSE END as offer_hire
--     , days_after_start
--   FROM combine_rows_to_coid
-- )
-- , create_days_behind as (
--   SELECT 
--       row_id 
--     , coid
--     , case 
--         when app_screen and screen_int and int_review and review_offer and offer_hire then days_after_start + days_app_screen + days_screen_int + days_int_review + days_review_offer + days_offer_hire
--         when app_screen and screen_int and int_review and review_offer then days_after_start + days_app_screen + days_screen_int + days_int_review + days_review_offer
--         when app_screen and screen_int and int_review then days_after_start + days_app_screen + days_screen_int + days_int_review
--         when app_screen and screen_int then days_after_start + days_app_screen + days_screen_int
--         when app_screen then days_after_start + days_app_screen
--         else NULL
--       end as days_app_screen
--     , app_screen 
--     , case 
--         when screen_int and int_review and review_offer and offer_hire then days_after_start + days_screen_int + days_int_review + days_review_offer + days_offer_hire
--         when screen_int and int_review and review_offer then days_after_start + days_screen_int + days_int_review + days_review_offer
--         when screen_int and int_review then days_after_start + days_screen_int + days_int_review
--         when screen_int then days_after_start + days_screen_int
--         else days_after_start 
--       end as days_screen_int
--     , screen_int
--     , case 
--         when int_review and review_offer and offer_hire then days_after_start + days_int_review + days_review_offer + days_offer_hire
--         when int_review and review_offer then days_after_start + days_int_review + days_review_offer
--         when int_review then days_after_start + days_int_review
--         else NULL 
--       end as days_int_review
--     , int_review
--     , case 
--         when review_offer and offer_hire then days_after_start + days_review_offer + days_offer_hire
--         when review_offer then days_after_start + days_review_offer
--         else NULL 
--       end as days_review_offer
--     , review_offer
--     , case 
--         when offer_hire then days_after_start + days_offer_hire
--         else NULL 
--       end as days_offer_hire
--     , offer_hire
--   FROM create_outcomes
-- )
-- SELECT 
--     row_id 
--   , coid 
--   , app_screen
--   , screen_int
--   , int_review
--   , review_offer
--   , offer_hire 
--   , date_add('2023-01-31', interval cast(((days_app_screen * -1)) + (rand()*-15) as int64) day) as date_application
--   , date_add('2023-01-31', interval cast(days_app_screen * -1 as int64) day) as date_screen
--   , date_add('2023-01-31', interval cast(days_screen_int * -1 as int64) day) as date_interview
--   , date_add('2023-01-31', interval cast(days_int_review * -1 as int64) day) as date_review
--   , date_add('2023-01-31', interval cast(days_review_offer * -1 as int64) day) as date_offer
--   , date_add('2023-01-31', interval cast(days_offer_hire * -1 as int64) day) as date_hire
-- FROM create_days_behind
-- ORDER BY 1 
-- ;

-- ### Create a snapshot by day

-- CREATE OR REPLACE TABLE `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_recruiting_pipeline_fake_data_daily_snapshot`  AS 
-- with rows10 as (
--             SELECT 1 as counter_id
--   UNION ALL SELECT 2
--   UNION ALL SELECT 3
--   UNION ALL SELECT 4
--   UNION ALL SELECT 5
--   UNION ALL SELECT 6
--   UNION ALL SELECT 7
--   UNION ALL SELECT 8
--   UNION ALL SELECT 9
--   UNION ALL SELECT 10
-- )
-- , rows365 as (
--   SELECT row_number() over (order by 'x') as days_since_app 
--   FROM rows10 a -- 10
--   , rows10 b -- 100 
--   , rows10 c -- 1k
--   LIMIT 365
-- )
-- , every_stage_by_row as (
--   SELECT 
--       row_id 
--     , date_diff(date_application, date_application, day) as days_to_app
--     , date_diff(date_screen, date_application, day) as days_to_screen
--     , date_diff(date_interview, date_application, day) as days_to_int
--     , date_diff(date_review, date_application, day) as days_to_review
--     , date_diff(date_offer, date_application, day) as days_to_offer
--     , date_diff(date_hire, date_application, day) as days_to_hire
--   FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_recruiting_pipeline_fake_data`
-- )
-- , max_until_out as (
--   SELECT 
--       *
--     , case 
--         when days_to_screen is null then days_to_app
--         when days_to_int is null then days_to_screen
--         when days_to_review is null then days_to_int
--         when days_to_offer is null then days_to_review
--         when days_to_hire is null then days_to_offer
--         else days_to_hire
--       end as max_days_until_out 
--   FROM every_stage_by_row
-- )
-- , combine_365_with_every_stage as (
--   SELECT 
--       b.row_id 
--     , a.days_since_app
--     , floor(a.days_since_app / (365/12)) as months_since_app
--     , case 
--         when a.days_since_app > b.max_days_until_out then NULL
--         when a.days_since_app >= b.days_to_app and a.days_since_app < coalesce(b.days_to_screen,999) then 'F Application Complete'
--         when a.days_since_app >= b.days_to_screen and a.days_since_app < coalesce(b.days_to_int,999) then 'E Screen Complete'
--         when a.days_since_app >= b.days_to_int and a.days_since_app < coalesce(b.days_to_review,999) then 'D Interview Complete'
--         when a.days_since_app >= b.days_to_review and a.days_since_app < coalesce(b.days_to_offer,999) then 'C Review Complete'
--         when a.days_since_app >= b.days_to_offer and a.days_since_app < coalesce(b.days_to_hire,999) then 'B Offer Sent'
--         when a.days_since_app >= b.days_to_hire then 'A Hired'
--         else 'Unknown'
--       end as status
--   FROM rows365 a 
--   , max_until_out b
-- )
-- SELECT 
--   * 
--   --   row_id
--   -- , months_since_app
--   -- , min(status) as status
-- -- FROM combine_365_with_every_stage a 
-- FROM max_until_out
-- WHERE row_id = 2 
-- -- GROUP BY 1,2
-- ORDER BY 1,2
-- ;

-- /*******************
-- Section 1 extra: Add fake data to this table
-- *******************/

-- CREATE OR REPLACE TABLE `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_1_month_fake_data`  AS 
-- ### Add in target headcount -- target_headcount
-- with add_target_headcount as (
--   SELECT 
--       *
--     , round(total_headcount * (1.25 + (rand()/10)),0) as total_target_headcount
--   FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_1_month_widen`
-- )
-- ### Add 12 months of data
-- , add_12_months as (
--             SELECT 0 as months_ago 
--   UNION ALL SELECT 1
--   UNION ALL SELECT 2
--   UNION ALL SELECT 3
--   UNION ALL SELECT 4
--   UNION ALL SELECT 5
--   UNION ALL SELECT 6
--   UNION ALL SELECT 7
--   UNION ALL SELECT 8
--   UNION ALL SELECT 9
--   UNION ALL SELECT 10
--   UNION ALL SELECT 11
-- )
-- SELECT 
--   * except(
--         row_id
--       , measurement_date
--       , total_headcount
--       , total_hires
--       , total_terms
--       , total_skillmix_in
--       , total_skillmix_out
--       , total_status_change_in
--       , total_status_change_out
--       , total_target_headcount
--     )
--   , (row_id * months_ago * 500) + cast(1234561345 * rand() as int64) as row_id
--   , date_add(measurement_date, INTERVAL months_ago * -1 month) as measurement_date 
--   , round(total_headcount * (1 + (rand()/10)),0) as total_headcount
--   , round(total_hires * (1 + (rand()/10)),0) as total_hires
--   , round(total_terms * (1 + (rand()/10)),0) as total_terms
--   , round(total_skillmix_in * ((rand()/10)),0) as total_skillmix_in
--   , round(total_skillmix_out * (1 + (rand()/10)),0) as total_skillmix_out
--   , round(total_status_change_in * (1 + (rand()/10)),0) as total_status_change_in
--   , round(total_status_change_out * (1 + (rand()/10)),0) as total_status_change_out
--   , round(total_target_headcount * (1 + (rand()/10)),0) as total_target_headcount
-- FROM add_target_headcount a 
-- , add_12_months 
-- ORDER BY row_id 
-- ;

/*
### Create another table w/ entity_date_id, status, total_headcount_bom, total_headcount_eom, total_target_headcount

CREATE OR REPLACE TABLE `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_1_month_widen_headcount`  AS 
SELECT 
    entity_status_date_id 
  , total_headcount 
FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_1_month_widen`
WHERE total_headcount is not null
; 

CREATE OR REPLACE TABLE `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_1_month_widen_main`  AS 
SELECT *
FROM `hca-sandbox-aaron-argolis.hrg.pipeline_health_metrics_sample_1_month_widen`
WHERE total_headcount is null 
ORDER BY 1 
; 
*/

-- -- Table 2
-- -- Application ID
-- -- Facility
-- -- Function
-- -- Sub-function
-- -- Nurse type (new grad vs experienced)
-- -- Application date
-- -- Screen date
-- -- Interviews date
-- -- Review date
-- -- Extend Offer date
-- -- Accept / reject date (and decision)
-- -- Hire date