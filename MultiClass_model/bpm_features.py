import pandas as pd
import numpy as np
import datetime
import subprocess
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta




def calc_all_features(spark, target, tables_dict: dict, model_name: str, output_scheme: str, mode: str, oot_month = None):
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.sql import DataFrame
    from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation
    
    target.createOrReplaceTempView('target')
    start_ = target.select(F.min(F.col('report_dt'))).toPandas()['min(report_dt)'][0]
    end_ = target.select(F.max(F.col('report_dt'))).toPandas()['max(report_dt)'][0]
    dates = pd.date_range(start = start_, end = end_, freq = pd.offsets.MonthEnd()).strftime("%Y-%m-%d").tolist()

    def fix_spark_types(df):

        for col, dtype in tqdm(df.dtypes):
                if "decimal" in dtype:
                    df = df.withColumn(col, F.col(col).cast(T.DoubleType()))

        date_columns = [i for i in df.columns if '_dt' in i]
        for col in tqdm(date_columns):
            df = df.withColumn(col, F.when(F.col(col) > F.to_date(F.lit(pd.Timestamp.max)), F.to_date(F.lit(pd.Timestamp.max))).otherwise(F.col(col))) 

        return df
    
    def _terminal_ld_3_0_bash_run(bashCommand):
        process = subprocess.Popen(bashCommand, stdout=subprocess.PIPE, shell=True)
        process.communicate()
        
    def get_diff_dates(report_dt: str) -> tuple:

        current_date = datetime.strptime(report_dt, '%Y-%m-%d')

        date_3m = current_date - relativedelta(months = +2)
        next_mnth_3 = date_3m.replace(day=28) + timedelta(days = 4)
        date_3m = datetime.strftime(next_mnth_3 - timedelta(days=next_mnth_3.day), '%Y-%m-%d')
        date_3m_1 = datetime.strftime(datetime.strptime(date_3m, '%Y-%m-%d').replace(day = 1), '%Y-%m-%d')



        date_6m = current_date - relativedelta(months = +5)
        next_mnth_6 = date_6m.replace(day=28) + timedelta(days = 4)
        date_6m = datetime.strftime(next_mnth_6 - timedelta(days=next_mnth_6.day), '%Y-%m-%d')
        date_6m_1 = datetime.strftime(datetime.strptime(date_6m, '%Y-%m-%d').replace(day = 1), '%Y-%m-%d')



        date_12m = current_date - relativedelta(months = +11)
        next_mnth_12 = date_12m.replace(day=28) + timedelta(days = 4)
        date_12m = datetime.strftime(next_mnth_12 - timedelta(days=next_mnth_12.day), '%Y-%m-%d')
        date_12m_1 = datetime.strftime(datetime.strptime(date_12m, '%Y-%m-%d').replace(day = 1), '%Y-%m-%d')

        return date_3m, date_3m_1, date_6m, date_6m_1, date_12m, date_12m_1

    def check_table_exist(scheme,table_name):
        table_exist = False
        try:
            spark.read.table(f"{scheme}.{table_name}")
            table_exist = True
        except:
            pass
        return table_exist
    
    def remove_correlated_features(df, threshold: float = 0.8, method: str = 'pearson'):
        

        # 1. Выбор числовых столбцов
        numeric_types = (DoubleType, FloatType, IntegerType, LongType)
        numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, numeric_types)]

        # Если числовых столбцов меньше 2, преобразования не нужны
        if len(numeric_cols) < 2:
            return df

        # 2. Вычисление матрицы корреляций
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol='features_vector')
        df_vector = assembler.transform(df).select('features_vector')
        corr_matrix = Correlation.corr(df_vector, 'features_vector', method).head()[0].toArray()

        # 3. Определение столбцов для удаления
        to_remove = set()
        n = len(numeric_cols)

        for i in range(n):
            if numeric_cols[i] in to_remove:
                continue
            for j in range(i + 1, n):
                if abs(corr_matrix[i][j]) > threshold:
                    to_remove.add(numeric_cols[j])

        # 4. Удаление столбцов и возврат результата
        return df.drop(*to_remove)

    
    def agg_features():
        agg = (spark.read.table(tables_dict['agg'])  
                    .where(F.col('report_dt').isin(dates))
              )

        agg = (agg.withColumn('report_dt', F.to_date('report_dt'))
                  .join(F.broadcast(target.select('epk_id', 'report_dt')),
                        how = 'inner',
                        on = ['epk_id', 'report_dt'],
                        )
              )
        if oot_month is None:
            if not check_table_exist(output_scheme, f'{model_name}_agg_features_train'):
                agg.write.saveAsTable(f"{output_scheme}.{model_name}_agg_features_train", mode = 'overwrite')
        else:
            if not check_table_exist(output_scheme, f'{model_name}_agg_features_train'):
                agg.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_agg_features_train", mode = 'overwrite')
            if not check_table_exist(output_scheme, f'{model_name}_agg_features_oot'):
                agg.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_agg_features_oot", mode = 'overwrite')
        print('agg_features: done')
        
        
    def arrests_features():
        e_cod = spark.read.table(tables_dict['e_cod']).select('printableNo','epkId').dropDuplicates()
        idoc = spark.read.table(tables_dict['idoc'])
        idoc_acc = spark.read.table(tables_dict['idoc_acc'])
        idoc_and_acc = idoc.join(idoc_acc,
                         (idoc_acc.idoc_major == idoc.id_major) &
                         (idoc_acc.idoc_minor == idoc.id_minor) &
                         (idoc_acc.id_mega == idoc.id_mega)
                        ).dropDuplicates()
        
        idoc_and_acc_and_cod = idoc_and_acc.join(e_cod,
                                         e_cod.printableNo == idoc_and_acc.id_account)
        idoc_and_acc_and_cod.createOrReplaceTempView('idoc_and_acc_and_cod')
        querry2 = ''
        for col in ['sum_val_aressts','qty_val_aressts']:
            querry2 = querry2 + f'''
                CAST(avg(coalesce(agg_1m.{col},0)) AS double) as {col}_avg_1m,
                CAST(avg(coalesce(agg_3m.{col},0)) AS double) as {col}_avg_3m,
                CAST(avg(coalesce(agg_6m.{col},0)) AS double) as {col}_avg_6m,
                CAST(avg(coalesce(agg_9m.{col},0)) AS double) as {col}_avg_9m,
                CAST(avg(coalesce(agg_12m.{col},0)) AS double) as {col}_avg_12m,
                CAST(avg(coalesce(agg_3m.{col},0)) / avg(coalesce(agg_6m.{col},0)) AS double) AS {col}_avg_3_6,
                CAST(avg(coalesce(agg_3m.{col},0)) / avg(coalesce(agg_9m.{col},0)) AS double) AS {col}_avg_3_9,
                CAST(avg(coalesce(agg_3m.{col},0)) / avg(coalesce(agg_12m.{col},0)) AS double) AS {col}_avg_3_12,
                CAST(avg(coalesce(agg_6m.{col},0)) / avg(coalesce(agg_9m.{col},0)) AS double) AS {col}_avg_6_9,
                CAST(avg(coalesce(agg_6m.{col},0)) / avg(coalesce(agg_12m.{col},0)) AS double) AS {col}_avg_6_12,
                CAST(avg(coalesce(agg_9m.{col},0)) / avg(coalesce(agg_12m.{col},0)) AS double) AS {col}_avg_9_12, 
                CAST(sum(coalesce(agg_1m.{col},0)) AS double) as {col}_sum_1m,
                CAST(sum(coalesce(agg_3m.{col},0)) AS double) as {col}_sum_3m,
                CAST(sum(coalesce(agg_6m.{col},0)) AS double) as {col}_sum_6m,
                CAST(sum(coalesce(agg_9m.{col},0)) AS double) as {col}_sum_9m,
                CAST(sum(coalesce(agg_12m.{col},0)) AS double) as {col}_sum_12m,
                CAST(sum(coalesce(agg_3m.{col},0)) / sum(coalesce(agg_6m.{col},0)) AS double) AS {col}_sum_3_6,
                CAST(sum(coalesce(agg_3m.{col},0)) / sum(coalesce(agg_9m.{col},0)) AS double) AS {col}_sum_3_9,
                CAST(sum(coalesce(agg_3m.{col},0)) / sum(coalesce(agg_12m.{col},0)) AS double) AS {col}_sum_3_12,
                CAST(sum(coalesce(agg_6m.{col},0)) / sum(coalesce(agg_9m.{col},0)) AS double) AS {col}_sum_6_9,
                CAST(sum(coalesce(agg_6m.{col},0)) / sum(coalesce(agg_12m.{col},0)) AS double) AS {col}_sum_6_12,
                CAST(sum(coalesce(agg_9m.{col},0)) / sum(coalesce(agg_12m.{col},0)) AS double) AS {col}_sum_9_12,'''

        arrests = spark.sql(f'''
        WITH arrests AS(
            SELECT
                last_day(id_docdate) as report_dt, 
                epkId as epk_id, 
                sum(d_sum_val) as sum_val_aressts,
                count(CASE WHEN d_sum_val > 0 THEN d_sum_val ELSE NULL END) as qty_val_aressts
            FROM
                idoc_and_acc_and_cod
            WHERE
                last_day(id_docdate) BETWEEN last_day(add_months('{start_}', -11)) AND '{end_}'
            GROUP BY
                last_day(id_docdate),
                epkId
        ), 
        target_deep as (
            SELECT
                epk_id,
                report_dt,
                last_day(add_months(report_dt, -2)) AS report_dt_3m,
                last_day(add_months(report_dt, -5)) AS report_dt_6m,
                last_day(add_months(report_dt, -8)) AS report_dt_9m,
                last_day(add_months(report_dt, -11)) AS report_dt_12m
            FROM 
                target   
        ), 
        arrest_deep as (
            SELECT DISTINCT
                target_deep.epk_id,
                target_deep.report_dt,
                {querry2[:-1]}
            FROM
                target_deep
                LEFT JOIN arrests agg_1m USING (epk_id)
                LEFT JOIN arrests agg_3m USING (epk_id)
                LEFT JOIN arrests agg_6m USING (epk_id)
                LEFT JOIN arrests agg_9m USING (epk_id)
                LEFT JOIN arrests agg_12m USING (epk_id)
            WHERE
                agg_1m.report_dt = target_deep.report_dt 
                AND agg_3m.report_dt BETWEEN target_deep.report_dt_3m AND target_deep.report_dt
                AND agg_6m.report_dt BETWEEN target_deep.report_dt_6m AND target_deep.report_dt
                AND agg_9m.report_dt BETWEEN target_deep.report_dt_9m AND target_deep.report_dt
                AND agg_12m.report_dt BETWEEN target_deep.report_dt_12m AND target_deep.report_dt
            GROUP BY
                target_deep.report_dt,
                target_deep.epk_id    
        )
        SELECT * FROM arrest_deep
        ''')
        
        
        if oot_month is None:
            if not check_table_exist(output_scheme, f'{model_name}_arrests_features_train'):
                arrests.write.saveAsTable(f"{output_scheme}.{model_name}_arrests_features_train", mode = 'overwrite')
        else:
            if not check_table_exist(output_scheme, f'{model_name}_arrests_features_train'):
                arrests.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_arrests_features_train", mode = 'overwrite')
            if not check_table_exist(output_scheme, f'{model_name}_arrests_features_oot'):
                arrests.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_arrests_features_oot", mode = 'overwrite')
        print('arrests_features: done')

    def feedbacks_features():  
        feedbacks = spark.read.table(tables_dict['feedbacks']).withColumnRenamed('epk_id', 'epkid') 
        feedbacks_response_12 = (feedbacks
                                 .join(target.select('epk_id', 'report_dt'),
                                       on = [feedbacks.epkid == target.epk_id, 
                                             F.to_date(feedbacks.report_dt) <= F.to_date(target.report_dt),
                                             F.to_date(feedbacks.report_dt) >= F.add_months(F.to_date(target.report_dt), -11)
                                            ],
                                       how = 'right'
                                      )
                                ).fillna(0) \
                                 .groupby('epk_id', target.report_dt)\
                                 .agg(F.sum('opened_feedbacks_cnt').alias('pfm_opened_cnt_all_12m'),
                                      F.sum('started_feedbacks_cnt').alias('pfm_started_cnt_all_12m'),
                                      F.sum('liked_feedbacks_cnt').alias('pfm_like_cnt_all_12m'),
                                      F.sum('opened_invest_feedbacks_cnt').alias('pfm_opened_invest_cnt_all_12m'),
                                      F.sum('started_invest_feedbacks_cnt').alias('pfm_started_invest_cnt_all_12m'),
                                      F.sum('liked_invest_feedbacks_cnt').alias('pfm_like_cnt_invest_all_12m'),
                                     )

        feedbacks_response_3 = (feedbacks
                                 .join(target.select('epk_id', 'report_dt'),
                                       on = [feedbacks.epkid == target.epk_id, 
                                             F.to_date(feedbacks.report_dt) <= F.to_date(target.report_dt),
                                             F.to_date(feedbacks.report_dt) >= F.add_months(F.to_date(target.report_dt), -2)
                                            ],
                                       how = 'right'
                                      )
                                ).fillna(0) \
                                 .groupby('epk_id', target.report_dt)\
                                 .agg(F.sum('opened_feedbacks_cnt').alias('pfm_opened_cnt_all_3m'),
                                      F.sum('started_feedbacks_cnt').alias('pfm_started_cnt_all_3m'),
                                      F.sum('liked_feedbacks_cnt').alias('pfm_like_cnt_all_3m'),
                                      F.sum('opened_invest_feedbacks_cnt').alias('pfm_opened_invest_cnt_all_3m'),
                                      F.sum('started_invest_feedbacks_cnt').alias('pfm_started_invest_cnt_all_3m'),
                                      F.sum('liked_invest_feedbacks_cnt').alias('pfm_like_invest_cnt_all_3m'),
                                     )

        feedbacks_response = (feedbacks_response_12.join(feedbacks_response_3,
                                                        on = ['epk_id', 'report_dt'],
                                                        how = 'left')
                                                  .join(target.select('epk_id', 'report_dt'),
                                                        how = 'inner',
                                                        on = ['epk_id', 'report_dt'],
                                                        )
                             )                    

        if oot_month is None:
            if not check_table_exist(output_scheme, f'{model_name}_feedbacks_features_train'):
                feedbacks_response.write.saveAsTable(f"{output_scheme}.{model_name}_feedbacks_features_train", mode = 'overwrite')
        else:
            if not check_table_exist(output_scheme, f'{model_name}_feedbacks_features_train'):
                feedbacks_response.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_feedbacks_features_train", mode = 'overwrite')
            if not check_table_exist(output_scheme, f'{model_name}_feedbacks_features_oot'):
                feedbacks_response.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_feedbacks_features_oot", mode = 'overwrite')
        print('feedbacks_features: done')

    def vsp_visits_features():
        def vsp_visits(dt, months):
            df = spark.sql(f'''
            select 
                epk_id,
                '{dt}' as report_dt,
                count(visit_id) as vsp_visit_qty_{months}m
            from 
                {tables_dict['vsp_visits']}
            where 
                day_part between (date('{dt}') - interval '{months}' month) and date('{dt}')
            group by
                epk_id,
                report_dt
        ''')
            return df

        for report_dt in dates:
            if report_dt == dates[0]:
                vsp_feats_3m  = vsp_visits(report_dt,'3')
                vsp_feats_12m  = vsp_visits(report_dt,'12')
            else:
                vsp_feats_3m.union(vsp_visits(report_dt,'3'))
                vsp_feats_3m.union(vsp_visits(report_dt,'12'))

        vsp_feats = (vsp_feats_12m.join(vsp_feats_3m,
                                        on = ['epk_id', 'report_dt'],
                                        how = 'left')
                                  .join(target.select('epk_id', 'report_dt'),
                                        how = 'inner',
                                        on = ['epk_id', 'report_dt'],
                                        )
                    )

        if oot_month is None:
            if not check_table_exist(output_scheme, f'{model_name}_vsp_visits_features_train'):
                vsp_feats.write.saveAsTable(f"{output_scheme}.{model_name}_vsp_visits_features_train", mode = 'overwrite')
        else:
            if not check_table_exist(output_scheme, f'{model_name}_vsp_visits_features_train'):
                vsp_feats.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_vsp_visits_features_train", mode = 'overwrite')
            if not check_table_exist(output_scheme, f'{model_name}_vsp_visits_features_oot'):
                vsp_feats.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_vsp_visits_features_oot", mode = 'overwrite')
        print('vsp_visits_features: done')

    def card_transactions_features():
        card_transactions = (spark.read.table(tables_dict['card_transactions'])  
                                        .where(F.col('report_dt').isin(dates))
              )

        card_transactions = (card_transactions.withColumn('report_dt', F.to_date('report_dt'))
                                              .join(target.select('epk_id', 'report_dt'),
                                                    how = 'inner',
                                                    on = ['epk_id', 'report_dt'],
                                                    )
              )
        if oot_month is None:
            if not check_table_exist(output_scheme, f'{model_name}_card_transactions_features_train'):
                card_transactions.write.saveAsTable(f"{output_scheme}.{model_name}_card_transactions_features_train", mode = 'overwrite')
        else:
            if not check_table_exist(output_scheme, f'{model_name}_card_transactions_features_train'):
                card_transactions.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_card_transactions_features_train", mode = 'overwrite')
            if not check_table_exist(output_scheme, f'{model_name}_card_transactions_features_oot'):
                card_transactions.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_card_transactions_features_oot", mode = 'overwrite')
        print('card_transactions_features: done')

    def card_transactions_10d_features():
        card_transactions = (spark.read.table(tables_dict['card_transactions_10d'])  
                                        .where(F.col('report_dt').isin(dates))
              )

        card_transactions = (card_transactions.withColumn('report_dt', F.to_date('report_dt'))
                                              .join(target.select('epk_id', 'report_dt'),
                                                    how = 'inner',
                                                    on = ['epk_id', 'report_dt'],
                                                    )
              )
        if oot_month is None:
            if not check_table_exist(output_scheme, f'{model_name}_card_transactions_10d_features_train'):
                card_transactions.write.saveAsTable(f"{output_scheme}.{model_name}_card_transactions_10d_features_train", mode = 'overwrite')
        else:
            if not check_table_exist(output_scheme, f'{model_name}_card_transactions_features_train'):
                card_transactions.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_card_transactions_10d_features_train", mode = 'overwrite')
            if not check_table_exist(output_scheme, f'{model_name}_card_transactions_10d_features_oot'):
                card_transactions.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_card_transactions_10d_features_oot", mode = 'overwrite')
        print('card_transactions_features_10d: done')
        
    def flows12_features():
        flows12 = spark.sql(f"""
        WITH flows AS(
            SELECT
                epk_id, 
                report_dt,
                crd_inf_total_rub_amt, --Σ поступлений по всем картам
                crd_inf_total_qty, -- N поступлений по всем картам
                crd_dc_tot_bal_rub_amt, -- Общий баланс по всем действующим ДК
                crd_otf_pos_spend_rub_amt, -- Σ покупок (POS) по всем картам
                crd_otf_pos_spend_qty, -- N покупок (POS) по всем картам
                crd_otf_total_rub_amt, -- Σ списаний по всем картам
                crd_otf_total_qty, --N списаний по всем картам
                dep_inf_income_rub_amt, --Σ поступлений на счет (зарплата, пенсии, больничные)
                dep_otf_total_rub_amt, -- Σ списаний со всех счетов
                dep_otf_total_qty, --N списаний со всех счетов
                pnl_oi_ca_amt, --Σ ОД по текущим счетам
                pnl_oi_cc_amt, --Σ ОД по КК
                pnl_oi_dc_amt, --Σ ОД по ДК
                pnl_oi_ml_amt, --Σ ОД по ипотечным кредитам
                pnl_oi_othr_amt, --Σ ОД по прочим продуктам
                pnl_oi_pl_amt, --Σ ОД по потребительским кредитам
                pnl_oi_td_amt, --Σ ОД по срочным счетам
                pnl_oi_total_amt, --Σ общего ОД
                srv_m2m_inf_other_bank_amt, --Σ входящих переводов на свои счета из других банков
                srv_m2m_inf_other_bank_qty, --N входящих переводов на свои счета из других банков
                srv_pmt_all_amt, -- Σ всех платежей через раздел Платежи СБОЛ (включая АП)
                srv_pmt_all_qty, -- N всех платежей через раздел Платежи СБОЛ (включая АП)
                crd_dc_m2m_other_bank_amt, -- Σ сдергиваний за 12м
                crd_inf_cash_adv_rub_amt, --Σ взносов наличных по всем картам
                crd_dc_p2p_other_bank_amt, --Σ переводов в другие банки по ДК
                crd_dc_p2p_other_bank_qty, --N переводов в другие банки по ДК,
                dep_acct_dep_td_bal_rub_amt, --срочные вклады
                dep_tot_exp_2m_bal_rub_amt, 
                dep_tot_exp_3m_bal_rub_amt, 
                prd_da_active_nflag,
                dep_acct_dep_tot_bal_rub_amt,
                dep_acct_dep_tot_qty, 
                dep_inf_income_rub_amt / nullif(crd_inf_total_rub_amt, 0) as in_payroll_share,
                crd_otf_cash_rub_amt / nullif(crd_inf_total_rub_amt, 0) as out_cash_share,
                crd_otf_pos_spend_rub_amt / nullif(crd_inf_total_rub_amt, 0) as out_pos_share,
                crd_otf_pos_spend_rub_amt / nullif(crd_dc_tot_bal_rub_amt, 0) as out_pos_to_balance_share,
                crd_otf_pos_spend_rub_amt / nullif(dep_otf_total_rub_amt, 0) as out_pos_to_spends_share,
                crd_dc_m2m_other_bank_amt / nullif(crd_inf_total_rub_amt, 0) as m2m_to_income_share,
                crd_dc_m2m_other_bank_amt / nullif(dep_otf_total_rub_amt, 0) as m2m_to_spends_share,
                crd_inf_cash_adv_rub_amt / nullif(crd_inf_total_rub_amt, 0) as in_cash_to_income_share,
                crd_dc_p2p_other_bank_amt / nullif(crd_inf_total_rub_amt, 0) as out_p2p_to_income_share,
                crd_dc_p2p_other_bank_amt / nullif(dep_otf_total_rub_amt, 0) as out_p2p_to_spends_share,
                dep_inf_income_qty / nullif(crd_inf_total_qty, 0) as in_payroll_share_qty,
                crd_otf_cash_qty / nullif(crd_inf_total_qty, 0) as out_cash_share_qty,
                crd_otf_pos_spend_qty / nullif(crd_inf_total_qty, 0) as out_pos_share_qty,
                crd_otf_pos_spend_qty / nullif(dep_otf_total_qty, 0) as out_pos_to_spends_share_qty,
                crd_dc_m2m_other_bank_qty / nullif(crd_inf_total_qty, 0) as m2m_to_income_share_qty,
                crd_dc_m2m_other_bank_qty / nullif(dep_otf_total_qty, 0) as m2m_to_spends_share_qty,
                crd_inf_cash_adv_qty / nullif(crd_inf_total_qty, 0) as in_cash_to_income_share_qty,
                crd_dc_p2p_other_bank_qty / nullif(crd_inf_total_qty, 0) as out_p2p_to_income_share_qty,
                crd_dc_p2p_other_bank_qty / nullif(dep_otf_total_qty, 0) as out_p2p_to_spends_share_qty,
                (coalesce(dep_tot_bal_rub_amt, 0) +
                            coalesce(inv_mf_agrmnt_bal_rub_amt, 0) +
                            coalesce(inv_tm_agrmnt_bal_rub_amt, 0) +
                            coalesce(inv_bo_agrmnt_bal_tot_rub_amt, 0) +
                            coalesce(bal_invest_insur_life_amt, 0) +
                            coalesce(bal_nakop_insur_life_amt, 0)
                            ) as tot_bal_with_invest
            FROM
                {tables_dict['agg']}
            WHERE
                report_dt BETWEEN last_day(add_months('{start_}', -11)) AND '{end_}'
        ),
        target_deep as (
            SELECT
                epk_id,
                report_dt,
                last_day(add_months(report_dt, -2)) AS report_dt_3m,
                last_day(add_months(report_dt, -11)) AS report_dt_12m
            FROM 
                target   
        ), 
        flows_deep as (
            SELECT 
                target_deep.epk_id,
                target_deep.report_dt,
                avg(agg_12m.crd_inf_total_rub_amt) as crd_inf_total_rub_amt_12m, --Σ поступлений по всем картам
                avg(agg_12m.crd_inf_total_qty) as crd_inf_total_qty_12m, -- N поступлений по всем картам
                avg(agg_12m.crd_dc_tot_bal_rub_amt) as crd_dc_tot_bal_rub_amt_12m, -- Общий баланс по всем действующим ДК
                avg(agg_12m.crd_otf_pos_spend_rub_amt) as crd_otf_pos_spend_rub_amt_12m, -- Σ покупок (POS) по всем картам
                avg(agg_12m.crd_otf_pos_spend_qty) as crd_otf_pos_spend_qty_12m, -- N покупок (POS) по всем картам
                avg(agg_12m.crd_otf_total_rub_amt) as crd_otf_total_rub_amt_12m, -- Σ списаний по всем картам
                avg(agg_12m.crd_otf_total_qty) as crd_otf_total_qty_12m, --N списаний по всем картам
                avg(agg_12m.dep_inf_income_rub_amt) as dep_inf_income_rub_amt_12, --Σ поступлений на счет (зарплата, пенсии, больничные)
                avg(agg_12m.dep_otf_total_rub_amt) as dep_otf_total_rub_amt_12m, -- Σ списаний со всех счетов
                avg(agg_12m.dep_otf_total_qty) as dep_otf_total_qty_12m, --N списаний со всех счетов
                avg(agg_12m.pnl_oi_ca_amt) as pnl_oi_ca_amt_12m, --Σ ОД по текущим счетам
                avg(agg_12m.pnl_oi_cc_amt) as pnl_oi_cc_amt_12m, --Σ ОД по КК
                avg(agg_12m.pnl_oi_dc_amt) as pnl_oi_dc_amt_12m, --Σ ОД по ДК
                avg(agg_12m.pnl_oi_ml_amt) as pnl_oi_ml_amt_12m, --Σ ОД по ипотечным кредитам
                avg(agg_12m.pnl_oi_othr_amt) as pnl_oi_othr_amt_12m, --Σ ОД по прочим продуктам
                avg(agg_12m.pnl_oi_pl_amt) as pnl_oi_pl_amt_12m, --Σ ОД по потребительским кредитам
                avg(agg_12m.pnl_oi_td_amt) as pnl_oi_td_amt_12m, --Σ ОД по срочным счетам
                avg(agg_12m.pnl_oi_total_amt) as pnl_oi_total_amt_12m, --Σ общего ОД
                avg(agg_12m.srv_m2m_inf_other_bank_amt) as srv_m2m_inf_other_bank_amt_12m, --Σ входящих переводов на свои счета из других банков
                avg(agg_12m.srv_m2m_inf_other_bank_qty) as srv_m2m_inf_other_bank_qty_12m, --N входящих переводов на свои счета из других банков
                avg(agg_12m.srv_pmt_all_amt) as srv_pmt_all_amt_12m, -- Σ всех платежей через раздел Платежи СБОЛ (включая АП)
                avg(agg_12m.srv_pmt_all_qty) as srv_pmt_all_qty_12m, -- N всех платежей через раздел Платежи СБОЛ (включая АП)
                avg(agg_12m.crd_dc_m2m_other_bank_amt) as crd_dc_m2m_other_bank_amt_12m, -- Σ сдергиваний за 12м
                avg(agg_12m.crd_inf_cash_adv_rub_amt) as crd_inf_cash_adv_rub_amt_12m, --Σ взносов наличных по всем картам
                avg(agg_12m.crd_dc_p2p_other_bank_amt) as crd_dc_p2p_other_bank_amt_12m, --Σ переводов в другие банки по ДК
                avg(agg_12m.crd_dc_p2p_other_bank_qty) as crd_dc_p2p_other_bank_qty_12m, --N переводов в другие банки по ДК,
                avg(agg_12m.dep_acct_dep_td_bal_rub_amt) as dep_acct_dep_td_bal_rub_amt_12m, --срочные вклады
                avg(agg_12m.dep_tot_exp_2m_bal_rub_amt) as dep_tot_exp_2m_bal_rub_amt_12m, 
                avg(agg_12m.dep_tot_exp_3m_bal_rub_amt) as dep_tot_exp_3m_bal_rub_amt_12m, 
                avg(agg_12m.prd_da_active_nflag) as prd_da_active_nflag_12m,
                avg(agg_12m.dep_acct_dep_tot_bal_rub_amt) as dep_acct_dep_tot_bal_rub_amt_12m,
                avg(agg_12m.dep_acct_dep_tot_qty) as dep_acct_dep_tot_qty_12m, 
                avg(agg_12m.in_payroll_share) as in_payroll_share_12m,
                avg(agg_12m.out_cash_share) as out_cash_share_12m,
                avg(agg_12m.out_pos_share) as out_pos_share_12m,
                avg(agg_12m.out_pos_to_balance_share) as out_pos_to_balance_share_12m,
                avg(agg_12m.out_pos_to_spends_share) as out_pos_to_spends_share_12m,
                avg(agg_12m.m2m_to_income_share) as m2m_to_income_share_12m,
                avg(agg_12m.m2m_to_spends_share) as m2m_to_spends_share_12m,
                avg(agg_12m.in_cash_to_income_share) as in_cash_to_income_share_12m,
                avg(agg_12m.out_p2p_to_income_share) as out_p2p_to_income_share_12m,
                avg(agg_12m.out_p2p_to_spends_share) as out_p2p_to_spends_share_12m,
                avg(agg_12m.in_payroll_share_qty) as in_payroll_share_qty_12m,
                avg(agg_12m.out_cash_share_qty) as out_cash_share_qty_12m,
                avg(agg_12m.out_pos_share_qty) as out_pos_share_qty_12m,
                avg(agg_12m.out_pos_to_spends_share_qty) as out_pos_to_spends_share_qty_12m,
                avg(agg_12m.m2m_to_income_share_qty) as m2m_to_income_share_qty_12m,
                avg(agg_12m.m2m_to_spends_share_qty) as m2m_to_spends_share_qty_12m,
                avg(agg_12m.in_cash_to_income_share_qty) as in_cash_to_income_share_qty_12m,
                avg(agg_12m.out_p2p_to_income_share_qty) as out_p2p_to_income_share_qty_12m,
                avg(agg_12m.out_p2p_to_spends_share_qty) as out_p2p_to_spends_share_qty_12m,
                avg(agg_12m.tot_bal_with_invest) as tot_bal_with_invest_12m,
                avg(agg_3m.tot_bal_with_invest) as tot_bal_with_invest_3m
            from 
                target_deep
                LEFT JOIN flows agg_3m USING (epk_id)
                LEFT JOIN flows agg_12m USING (epk_id)
            where 
                agg_3m.report_dt BETWEEN target_deep.report_dt_3m AND target_deep.report_dt
                AND agg_12m.report_dt BETWEEN target_deep.report_dt_12m AND target_deep.report_dt
            GROUP BY
                target_deep.report_dt,
                target_deep.epk_id
            )
            SELECT * FROM flows_deep
            """)



        if oot_month is None:
            if not check_table_exist(output_scheme, f'{model_name}_flows12_features_train'):
                flows12.write.saveAsTable(f"{output_scheme}.{model_name}_flows12_features_train", mode = 'overwrite')
        else:
            if not check_table_exist(output_scheme, f'{model_name}_flows12_features_train'):
                agg.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_flows12_features_train", mode = 'overwrite')
            if not check_table_exist(output_scheme, f'{model_name}_flows12_features_oot'):
                agg.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_flows12_features_oot", mode = 'overwrite')
        print('flows12_features: done')

    def pos_dynamic_features():
        pos_dynamic_features = spark.sql(f'''
        WITH
        target AS (
            SELECT
                report_dt,
                epk_id,
                last_day(add_months(report_dt, -2)) AS report_dt_3m,
                last_day(add_months(report_dt, -5)) AS report_dt_6m,
                last_day(add_months(report_dt, -8)) AS report_dt_9m,
                last_day(add_months(report_dt, -11)) AS report_dt_12m
            FROM target
        ),
        agg AS (
            SELECT
                report_dt,
                epk_id,
                coalesce(crd_dc_pos_auto_rub_amt, 0) + coalesce(crd_cc_pos_auto_rub_amt, 0) as pos_auto_rub_amt,
                coalesce(crd_dc_pos_beuaty_rub_amt, 0) + coalesce(crd_cc_pos_beuaty_rub_amt, 0) as pos_beuaty_rub_amt,
                coalesce(crd_dc_pos_clear_amt, 0) + coalesce(crd_cc_pos_clear_amt, 0) as pos_clear_amt,
                coalesce(crd_dc_pos_clothes_rub_amt, 0) + coalesce(crd_cc_pos_clothes_rub_amt, 0) as pos_clothes_rub_amt,
                coalesce(crd_dc_pos_eat_out_rub_amt, 0) + coalesce(crd_cc_pos_eat_out_rub_amt, 0) as pos_eat_out_rub_amt,
                coalesce(crd_dc_pos_groceries_rub_amt, 0) + coalesce(crd_cc_pos_groceries_rub_amt, 0) as pos_groceries_rub_amt,
                coalesce(crd_dc_pos_healthcare_rub_amt, 0) + coalesce(crd_cc_pos_healthcare_rub_amt, 0) as pos_healthcare_rub_amt,
                coalesce(crd_dc_pos_home_repair_rub_amt, 0) + coalesce(crd_cc_pos_home_repair_rub_amt, 0) as pos_home_repair_rub_amt,
                coalesce(crd_dc_pos_leisure_rub_amt, 0) + coalesce(crd_cc_pos_leisure_rub_amt, 0) as pos_leisure_rub_amt,
                coalesce(crd_dc_pos_othr_rub_amt, 0) + coalesce(crd_cc_pos_othr_rub_amt, 0) as pos_other_rub_amt, 
                coalesce(crd_dc_pos_money_trf_rub_amt, 0) + coalesce(crd_cc_pos_money_trf_rub_amt, 0) as pos_money_trf_rub_amt,
                coalesce(crd_dc_pos_dept_stores_rub_amt, 0) + coalesce(crd_cc_pos_dept_stores_rub_amt, 0) as pos_dept_stores_rub_amt,
                coalesce(crd_dc_pos_special_rub_amt, 0) + coalesce(crd_cc_pos_special_rub_amt, 0) as pos_special_rub_amt,
                coalesce(crd_dc_pos_pc_it_rub_amt, 0) + coalesce(crd_cc_pos_pc_it_rub_amt, 0) as pos_pc_it_rub_amt,
                coalesce(crd_dc_pos_telecom_rub_amt, 0) + coalesce(crd_cc_pos_telecom_rub_amt, 0) as pos_telecom_rub_amt,
                coalesce(crd_dc_pos_tourism_rub_amt, 0) + coalesce(crd_cc_pos_tourism_rub_amt, 0) as pos_tourism_rub_amt,
                coalesce(crd_dc_pos_utilities_rub_amt, 0) + coalesce(crd_cc_pos_utilities_rub_amt, 0) as pos_utilities_rub_amt,
                coalesce(srv_pmt_utl_amt, 0) as sbol_utilities_rub_amt,
                coalesce(crd_dc_pos_utilities_rub_amt, 0) + coalesce(crd_cc_pos_utilities_rub_amt, 0) + coalesce(srv_pmt_utl_amt, 0) as total_utilities_rub_amt
            FROM
                {tables_dict['agg']}
            WHERE
                report_dt BETWEEN
                    last_day(add_months('{start_}', -11))
                    AND '{end_}'
        ),
        features AS (
            SELECT
                /*+ BROADCAST(target) */
                target.report_dt,
                target.epk_id,

                CAST(avg(coalesce(agg_3m.pos_auto_rub_amt,0)) / avg(coalesce(agg_6m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_auto_rub_amt,0)) / avg(coalesce(agg_9m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_auto_rub_amt,0)) / avg(coalesce(agg_12m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_auto_rub_amt,0)) / avg(coalesce(agg_9m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_auto_rub_amt,0)) / avg(coalesce(agg_12m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_auto_rub_amt,0)) / avg(coalesce(agg_12m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_beuaty_rub_amt,0)) / avg(coalesce(agg_6m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_beuaty_rub_amt,0)) / avg(coalesce(agg_9m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_beuaty_rub_amt,0)) / avg(coalesce(agg_12m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_beuaty_rub_amt,0)) / avg(coalesce(agg_9m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_beuaty_rub_amt,0)) / avg(coalesce(agg_12m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_beuaty_rub_amt,0)) / avg(coalesce(agg_12m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_clear_amt,0)) / avg(coalesce(agg_6m.pos_clear_amt,0)) AS double) AS pos_clear_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_clear_amt,0)) / avg(coalesce(agg_9m.pos_clear_amt,0)) AS double) AS pos_clear_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_clear_amt,0)) / avg(coalesce(agg_12m.pos_clear_amt,0)) AS double) AS pos_clear_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_clear_amt,0)) / avg(coalesce(agg_9m.pos_clear_amt,0)) AS double) AS pos_clear_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_clear_amt,0)) / avg(coalesce(agg_12m.pos_clear_amt,0)) AS double) AS pos_clear_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_clear_amt,0)) / avg(coalesce(agg_12m.pos_clear_amt,0)) AS double) AS pos_clear_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_clothes_rub_amt,0)) / avg(coalesce(agg_6m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_clothes_rub_amt,0)) / avg(coalesce(agg_9m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_clothes_rub_amt,0)) / avg(coalesce(agg_12m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_clothes_rub_amt,0)) / avg(coalesce(agg_9m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_clothes_rub_amt,0)) / avg(coalesce(agg_12m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_clothes_rub_amt,0)) / avg(coalesce(agg_12m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_eat_out_rub_amt,0)) / avg(coalesce(agg_6m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_eat_out_rub_amt,0)) / avg(coalesce(agg_9m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_eat_out_rub_amt,0)) / avg(coalesce(agg_12m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_eat_out_rub_amt,0)) / avg(coalesce(agg_9m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_eat_out_rub_amt,0)) / avg(coalesce(agg_12m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_eat_out_rub_amt,0)) / avg(coalesce(agg_12m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_groceries_rub_amt,0)) / avg(coalesce(agg_6m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_groceries_rub_amt,0)) / avg(coalesce(agg_9m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_groceries_rub_amt,0)) / avg(coalesce(agg_12m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_groceries_rub_amt,0)) / avg(coalesce(agg_9m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_groceries_rub_amt,0)) / avg(coalesce(agg_12m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_groceries_rub_amt,0)) / avg(coalesce(agg_12m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_healthcare_rub_amt,0)) / avg(coalesce(agg_6m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_healthcare_rub_amt,0)) / avg(coalesce(agg_9m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_healthcare_rub_amt,0)) / avg(coalesce(agg_12m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_healthcare_rub_amt,0)) / avg(coalesce(agg_9m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_healthcare_rub_amt,0)) / avg(coalesce(agg_12m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_healthcare_rub_amt,0)) / avg(coalesce(agg_12m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_home_repair_rub_amt,0)) / avg(coalesce(agg_6m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_home_repair_rub_amt,0)) / avg(coalesce(agg_9m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_home_repair_rub_amt,0)) / avg(coalesce(agg_12m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_home_repair_rub_amt,0)) / avg(coalesce(agg_9m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_home_repair_rub_amt,0)) / avg(coalesce(agg_12m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_home_repair_rub_amt,0)) / avg(coalesce(agg_12m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_leisure_rub_amt,0)) / avg(coalesce(agg_6m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_leisure_rub_amt,0)) / avg(coalesce(agg_9m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_leisure_rub_amt,0)) / avg(coalesce(agg_12m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_leisure_rub_amt,0)) / avg(coalesce(agg_9m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_leisure_rub_amt,0)) / avg(coalesce(agg_12m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_leisure_rub_amt,0)) / avg(coalesce(agg_12m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_other_rub_amt,0)) / avg(coalesce(agg_6m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_other_rub_amt,0)) / avg(coalesce(agg_9m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_other_rub_amt,0)) / avg(coalesce(agg_12m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_other_rub_amt,0)) / avg(coalesce(agg_9m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_other_rub_amt,0)) / avg(coalesce(agg_12m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_other_rub_amt,0)) / avg(coalesce(agg_12m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_money_trf_rub_amt,0)) / avg(coalesce(agg_6m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_money_trf_rub_amt,0)) / avg(coalesce(agg_9m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_money_trf_rub_amt,0)) / avg(coalesce(agg_12m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_money_trf_rub_amt,0)) / avg(coalesce(agg_9m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_money_trf_rub_amt,0)) / avg(coalesce(agg_12m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_money_trf_rub_amt,0)) / avg(coalesce(agg_12m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_dept_stores_rub_amt,0)) / avg(coalesce(agg_6m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_dept_stores_rub_amt,0)) / avg(coalesce(agg_9m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_dept_stores_rub_amt,0)) / avg(coalesce(agg_12m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_dept_stores_rub_amt,0)) / avg(coalesce(agg_9m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_dept_stores_rub_amt,0)) / avg(coalesce(agg_12m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_dept_stores_rub_amt,0)) / avg(coalesce(agg_12m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_special_rub_amt,0)) / avg(coalesce(agg_6m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_special_rub_amt,0)) / avg(coalesce(agg_9m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_special_rub_amt,0)) / avg(coalesce(agg_12m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_special_rub_amt,0)) / avg(coalesce(agg_9m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_special_rub_amt,0)) / avg(coalesce(agg_12m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_special_rub_amt,0)) / avg(coalesce(agg_12m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_pc_it_rub_amt,0)) / avg(coalesce(agg_6m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_pc_it_rub_amt,0)) / avg(coalesce(agg_9m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_pc_it_rub_amt,0)) / avg(coalesce(agg_12m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_pc_it_rub_amt,0)) / avg(coalesce(agg_9m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_pc_it_rub_amt,0)) / avg(coalesce(agg_12m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_pc_it_rub_amt,0)) / avg(coalesce(agg_12m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_telecom_rub_amt,0)) / avg(coalesce(agg_6m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_telecom_rub_amt,0)) / avg(coalesce(agg_9m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_telecom_rub_amt,0)) / avg(coalesce(agg_12m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_telecom_rub_amt,0)) / avg(coalesce(agg_9m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_telecom_rub_amt,0)) / avg(coalesce(agg_12m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_telecom_rub_amt,0)) / avg(coalesce(agg_12m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_tourism_rub_amt,0)) / avg(coalesce(agg_6m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_tourism_rub_amt,0)) / avg(coalesce(agg_9m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_tourism_rub_amt,0)) / avg(coalesce(agg_12m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_tourism_rub_amt,0)) / avg(coalesce(agg_9m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_tourism_rub_amt,0)) / avg(coalesce(agg_12m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_tourism_rub_amt,0)) / avg(coalesce(agg_12m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.pos_utilities_rub_amt,0)) / avg(coalesce(agg_6m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.pos_utilities_rub_amt,0)) / avg(coalesce(agg_9m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.pos_utilities_rub_amt,0)) / avg(coalesce(agg_12m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.pos_utilities_rub_amt,0)) / avg(coalesce(agg_9m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.pos_utilities_rub_amt,0)) / avg(coalesce(agg_12m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.pos_utilities_rub_amt,0)) / avg(coalesce(agg_12m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.sbol_utilities_rub_amt,0)) / avg(coalesce(agg_6m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.sbol_utilities_rub_amt,0)) / avg(coalesce(agg_9m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.sbol_utilities_rub_amt,0)) / avg(coalesce(agg_12m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.sbol_utilities_rub_amt,0)) / avg(coalesce(agg_9m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.sbol_utilities_rub_amt,0)) / avg(coalesce(agg_12m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.sbol_utilities_rub_amt,0)) / avg(coalesce(agg_12m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_avg_9_12,

                CAST(avg(coalesce(agg_3m.total_utilities_rub_amt,0)) / avg(coalesce(agg_6m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_avg_3_6,
                CAST(avg(coalesce(agg_3m.total_utilities_rub_amt,0)) / avg(coalesce(agg_9m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_avg_3_9,
                CAST(avg(coalesce(agg_3m.total_utilities_rub_amt,0)) / avg(coalesce(agg_12m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_avg_3_12,
                CAST(avg(coalesce(agg_6m.total_utilities_rub_amt,0)) / avg(coalesce(agg_9m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_avg_6_9,
                CAST(avg(coalesce(agg_6m.total_utilities_rub_amt,0)) / avg(coalesce(agg_12m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_avg_6_12,
                CAST(avg(coalesce(agg_9m.total_utilities_rub_amt,0)) / avg(coalesce(agg_12m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_avg_9_12,



                --
                CAST(sum(coalesce(agg_3m.pos_auto_rub_amt,0)) / sum(coalesce(agg_6m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_auto_rub_amt,0)) / sum(coalesce(agg_9m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_auto_rub_amt,0)) / sum(coalesce(agg_12m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_auto_rub_amt,0)) / sum(coalesce(agg_9m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_auto_rub_amt,0)) / sum(coalesce(agg_12m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_auto_rub_amt,0)) / sum(coalesce(agg_12m.pos_auto_rub_amt,0)) AS double) AS pos_auto_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_beuaty_rub_amt,0)) / sum(coalesce(agg_6m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_beuaty_rub_amt,0)) / sum(coalesce(agg_9m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_beuaty_rub_amt,0)) / sum(coalesce(agg_12m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_beuaty_rub_amt,0)) / sum(coalesce(agg_9m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_beuaty_rub_amt,0)) / sum(coalesce(agg_12m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_beuaty_rub_amt,0)) / sum(coalesce(agg_12m.pos_beuaty_rub_amt,0)) AS double) AS pos_beuaty_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_clear_amt,0)) / sum(coalesce(agg_6m.pos_clear_amt,0)) AS double) AS pos_clear_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_clear_amt,0)) / sum(coalesce(agg_9m.pos_clear_amt,0)) AS double) AS pos_clear_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_clear_amt,0)) / sum(coalesce(agg_12m.pos_clear_amt,0)) AS double) AS pos_clear_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_clear_amt,0)) / sum(coalesce(agg_9m.pos_clear_amt,0)) AS double) AS pos_clear_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_clear_amt,0)) / sum(coalesce(agg_12m.pos_clear_amt,0)) AS double) AS pos_clear_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_clear_amt,0)) / sum(coalesce(agg_12m.pos_clear_amt,0)) AS double) AS pos_clear_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_clothes_rub_amt,0)) / sum(coalesce(agg_6m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_clothes_rub_amt,0)) / sum(coalesce(agg_9m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_clothes_rub_amt,0)) / sum(coalesce(agg_12m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_clothes_rub_amt,0)) / sum(coalesce(agg_9m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_clothes_rub_amt,0)) / sum(coalesce(agg_12m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_clothes_rub_amt,0)) / sum(coalesce(agg_12m.pos_clothes_rub_amt,0)) AS double) AS pos_clothes_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_eat_out_rub_amt,0)) / sum(coalesce(agg_6m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_eat_out_rub_amt,0)) / sum(coalesce(agg_9m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_eat_out_rub_amt,0)) / sum(coalesce(agg_12m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_eat_out_rub_amt,0)) / sum(coalesce(agg_9m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_eat_out_rub_amt,0)) / sum(coalesce(agg_12m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_eat_out_rub_amt,0)) / sum(coalesce(agg_12m.pos_eat_out_rub_amt,0)) AS double) AS pos_eat_out_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_groceries_rub_amt,0)) / sum(coalesce(agg_6m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_groceries_rub_amt,0)) / sum(coalesce(agg_9m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_groceries_rub_amt,0)) / sum(coalesce(agg_12m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_groceries_rub_amt,0)) / sum(coalesce(agg_9m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_groceries_rub_amt,0)) / sum(coalesce(agg_12m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_groceries_rub_amt,0)) / sum(coalesce(agg_12m.pos_groceries_rub_amt,0)) AS double) AS pos_groceries_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_healthcare_rub_amt,0)) / sum(coalesce(agg_6m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_healthcare_rub_amt,0)) / sum(coalesce(agg_9m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_healthcare_rub_amt,0)) / sum(coalesce(agg_12m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_healthcare_rub_amt,0)) / sum(coalesce(agg_9m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_healthcare_rub_amt,0)) / sum(coalesce(agg_12m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_healthcare_rub_amt,0)) / sum(coalesce(agg_12m.pos_healthcare_rub_amt,0)) AS double) AS pos_healthcare_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_home_repair_rub_amt,0)) / sum(coalesce(agg_6m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_home_repair_rub_amt,0)) / sum(coalesce(agg_9m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_home_repair_rub_amt,0)) / sum(coalesce(agg_12m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_home_repair_rub_amt,0)) / sum(coalesce(agg_9m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_home_repair_rub_amt,0)) / sum(coalesce(agg_12m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_home_repair_rub_amt,0)) / sum(coalesce(agg_12m.pos_home_repair_rub_amt,0)) AS double) AS pos_home_repair_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_leisure_rub_amt,0)) / sum(coalesce(agg_6m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_leisure_rub_amt,0)) / sum(coalesce(agg_9m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_leisure_rub_amt,0)) / sum(coalesce(agg_12m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_leisure_rub_amt,0)) / sum(coalesce(agg_9m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_leisure_rub_amt,0)) / sum(coalesce(agg_12m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_leisure_rub_amt,0)) / sum(coalesce(agg_12m.pos_leisure_rub_amt,0)) AS double) AS pos_leisure_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_other_rub_amt,0)) / sum(coalesce(agg_6m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_other_rub_amt,0)) / sum(coalesce(agg_9m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_other_rub_amt,0)) / sum(coalesce(agg_12m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_other_rub_amt,0)) / sum(coalesce(agg_9m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_other_rub_amt,0)) / sum(coalesce(agg_12m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_other_rub_amt,0)) / sum(coalesce(agg_12m.pos_other_rub_amt,0)) AS double) AS pos_other_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_money_trf_rub_amt,0)) / sum(coalesce(agg_6m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_money_trf_rub_amt,0)) / sum(coalesce(agg_9m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_money_trf_rub_amt,0)) / sum(coalesce(agg_12m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_money_trf_rub_amt,0)) / sum(coalesce(agg_9m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_money_trf_rub_amt,0)) / sum(coalesce(agg_12m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_money_trf_rub_amt,0)) / sum(coalesce(agg_12m.pos_money_trf_rub_amt,0)) AS double) AS pos_money_trf_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_dept_stores_rub_amt,0)) / sum(coalesce(agg_6m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_dept_stores_rub_amt,0)) / sum(coalesce(agg_9m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_dept_stores_rub_amt,0)) / sum(coalesce(agg_12m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_dept_stores_rub_amt,0)) / sum(coalesce(agg_9m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_dept_stores_rub_amt,0)) / sum(coalesce(agg_12m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_dept_stores_rub_amt,0)) / sum(coalesce(agg_12m.pos_dept_stores_rub_amt,0)) AS double) AS pos_dept_stores_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_special_rub_amt,0)) / sum(coalesce(agg_6m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_special_rub_amt,0)) / sum(coalesce(agg_9m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_special_rub_amt,0)) / sum(coalesce(agg_12m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_special_rub_amt,0)) / sum(coalesce(agg_9m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_special_rub_amt,0)) / sum(coalesce(agg_12m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_special_rub_amt,0)) / sum(coalesce(agg_12m.pos_special_rub_amt,0)) AS double) AS pos_special_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_pc_it_rub_amt,0)) / sum(coalesce(agg_6m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_pc_it_rub_amt,0)) / sum(coalesce(agg_9m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_pc_it_rub_amt,0)) / sum(coalesce(agg_12m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_pc_it_rub_amt,0)) / sum(coalesce(agg_9m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_pc_it_rub_amt,0)) / sum(coalesce(agg_12m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_pc_it_rub_amt,0)) / sum(coalesce(agg_12m.pos_pc_it_rub_amt,0)) AS double) AS pos_pc_it_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_telecom_rub_amt,0)) / sum(coalesce(agg_6m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_telecom_rub_amt,0)) / sum(coalesce(agg_9m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_telecom_rub_amt,0)) / sum(coalesce(agg_12m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_telecom_rub_amt,0)) / sum(coalesce(agg_9m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_telecom_rub_amt,0)) / sum(coalesce(agg_12m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_telecom_rub_amt,0)) / sum(coalesce(agg_12m.pos_telecom_rub_amt,0)) AS double) AS pos_telecom_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_tourism_rub_amt,0)) / sum(coalesce(agg_6m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_tourism_rub_amt,0)) / sum(coalesce(agg_9m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_tourism_rub_amt,0)) / sum(coalesce(agg_12m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_tourism_rub_amt,0)) / sum(coalesce(agg_9m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_tourism_rub_amt,0)) / sum(coalesce(agg_12m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_tourism_rub_amt,0)) / sum(coalesce(agg_12m.pos_tourism_rub_amt,0)) AS double) AS pos_tourism_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.pos_utilities_rub_amt,0)) / sum(coalesce(agg_6m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.pos_utilities_rub_amt,0)) / sum(coalesce(agg_9m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.pos_utilities_rub_amt,0)) / sum(coalesce(agg_12m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.pos_utilities_rub_amt,0)) / sum(coalesce(agg_9m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.pos_utilities_rub_amt,0)) / sum(coalesce(agg_12m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.pos_utilities_rub_amt,0)) / sum(coalesce(agg_12m.pos_utilities_rub_amt,0)) AS double) AS pos_utilities_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.sbol_utilities_rub_amt,0)) / sum(coalesce(agg_6m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.sbol_utilities_rub_amt,0)) / sum(coalesce(agg_9m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.sbol_utilities_rub_amt,0)) / sum(coalesce(agg_12m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.sbol_utilities_rub_amt,0)) / sum(coalesce(agg_9m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.sbol_utilities_rub_amt,0)) / sum(coalesce(agg_12m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.sbol_utilities_rub_amt,0)) / sum(coalesce(agg_12m.sbol_utilities_rub_amt,0)) AS double) AS sbol_utilities_rub_amt_sum_9_12,

                CAST(sum(coalesce(agg_3m.total_utilities_rub_amt,0)) / sum(coalesce(agg_6m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_sum_3_6,
                CAST(sum(coalesce(agg_3m.total_utilities_rub_amt,0)) / sum(coalesce(agg_9m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_sum_3_9,
                CAST(sum(coalesce(agg_3m.total_utilities_rub_amt,0)) / sum(coalesce(agg_12m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_sum_3_12,
                CAST(sum(coalesce(agg_6m.total_utilities_rub_amt,0)) / sum(coalesce(agg_9m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_sum_6_9,
                CAST(sum(coalesce(agg_6m.total_utilities_rub_amt,0)) / sum(coalesce(agg_12m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_sum_6_12,
                CAST(sum(coalesce(agg_9m.total_utilities_rub_amt,0)) / sum(coalesce(agg_12m.total_utilities_rub_amt,0)) AS double) AS total_utilities_rub_amt_sum_9_12

            FROM
                target
                LEFT JOIN agg agg_3m USING (epk_id)
                LEFT JOIN agg agg_6m USING (epk_id)
                LEFT JOIN agg agg_9m USING (epk_id)
                LEFT JOIN agg agg_12m USING (epk_id)
            WHERE
                agg_3m.report_dt BETWEEN target.report_dt_3m AND target.report_dt
                AND agg_6m.report_dt BETWEEN target.report_dt_6m AND target.report_dt
                AND agg_9m.report_dt BETWEEN target.report_dt_9m AND target.report_dt
                AND agg_12m.report_dt BETWEEN target.report_dt_12m AND target.report_dt
            GROUP BY
                target.report_dt,
                target.epk_id
        )

        SELECT * FROM features
        ''')

        if oot_month is None:
            if not check_table_exist(output_scheme, f'{model_name}_pos_dynamic_features_train'):
                pos_dynamic_features.write.saveAsTable(f"{output_scheme}.{model_name}_pos_dynamic_features_train", mode = 'overwrite')
        else:
            if not check_table_exist(output_scheme, f'{model_name}_pos_dynamic_features_train'):
                pos_dynamic_features.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_pos_dynamic_features_train", mode = 'overwrite')
            if not check_table_exist(output_scheme, f'{model_name}_pos_dynamic_features_oot'):
                pos_dynamic_features.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_pos_dynamic_features_oot", mode = 'overwrite')
        print('pos_dynamic_features: done')
        
        
        
    def aggr_dynamic_features():
        aggr = spark.read.table('prx_bpm_client_aggr_custom_rozn_client_aggr.ft_client_aggr_mnth')
        aggr_columns = aggr.columns
        aggr_dp_columns = []
        for col in aggr_columns:
            if (("crd" in col[0:3]) or ("dep" in col[0:3]) or ("srv" in col[0:3])):
                if (('3'not in col) 
                     and ('6'not in col) 
                     and ('9'not in col) 
                     and ('12'not in col) 
                     and ('dt'not in col) 
                     and ('id'not in col) 
                     and ('nflag'not in col)
                     and ('cd'not in col)
                     and ('dk'not in col)
                     and ('rate' not in col)
                     and ('1st' not in col)
                     and ('lst' not in col)):
                    aggr_dp_columns.append(col)
     
        n = len(aggr_dp_columns)
        part_table = 0
        for part in [aggr_dp_columns[:int(n/3)], aggr_dp_columns[int(n/3+1): int(2*n/3)] ,aggr_dp_columns[int(2*n/3):]]:
            querry1 = ''
            querry2 = ''
            part_table += 1
            for col in part:
                querry1 = querry1 + f'coalesce({col}, 0) as {col},'
                querry2 = querry2 + f'''
                            --avg(coalesce(agg_3m.{col},0)) / avg(coalesce(agg_6m.{col},0)) AS {col}_avg_3_6,
                            --avg(coalesce(agg_3m.{col},0)) / avg(coalesce(agg_9m.{col},0)) AS {col}_avg_3_9,
                            --avg(coalesce(agg_3m.{col},0)) / avg(coalesce(agg_12m.{col},0)) AS {col}_avg_3_12,
                            --avg(coalesce(agg_6m.{col},0)) / avg(coalesce(agg_9m.{col},0)) AS {col}_avg_6_9,
                            --avg(coalesce(agg_6m.{col},0)) / avg(coalesce(agg_12m.{col},0)) AS {col}_avg_6_12,
                            --avg(coalesce(agg_9m.{col},0)) / avg(coalesce(agg_12m.{col},0)) AS {col}_avg_9_12,        
                            CAST(sum(coalesce(agg_3m.{col},0)) AS double) as {col}_sum_3m,
                            CAST(sum(coalesce(agg_6m.{col},0)) AS double) as {col}_sum_6m,
                            CAST(sum(coalesce(agg_9m.{col},0)) AS double) as {col}_sum_9m,
                            CAST(sum(coalesce(agg_12m.{col},0)) AS double) as {col}_sum_12m,
                            CAST(sum(coalesce(agg_3m.{col},0)) / sum(coalesce(agg_6m.{col},0)) AS double) AS {col}_sum_3_6,
                            CAST(sum(coalesce(agg_3m.{col},0)) / sum(coalesce(agg_9m.{col},0)) AS double) AS {col}_sum_3_9,
                            CAST(sum(coalesce(agg_3m.{col},0)) / sum(coalesce(agg_12m.{col},0)) AS double) AS {col}_sum_3_12,
                            CAST(sum(coalesce(agg_6m.{col},0)) / sum(coalesce(agg_9m.{col},0)) AS double) AS {col}_sum_6_9,
                            CAST(sum(coalesce(agg_6m.{col},0)) / sum(coalesce(agg_12m.{col},0)) AS double) AS {col}_sum_6_12,
                            CAST(sum(coalesce(agg_9m.{col},0)) / sum(coalesce(agg_12m.{col},0)) AS double) AS {col}_sum_9_12,'''

            pos_dynamic_features = spark.sql(f'''
            WITH
            target AS (
                SELECT
                    report_dt,
                    epk_id,
                    last_day(add_months(report_dt, -2)) AS report_dt_3m,
                    last_day(add_months(report_dt, -5)) AS report_dt_6m,
                    last_day(add_months(report_dt, -8)) AS report_dt_9m,
                    last_day(add_months(report_dt, -11)) AS report_dt_12m
                FROM target
            ),
            agg AS (
                SELECT
                    report_dt,
                    epk_id,
                    {querry1[:-1]}
                FROM
                    {tables_dict['agg']}
                WHERE
                    report_dt BETWEEN
                        last_day(add_months('{start_}', -11))
                        AND '{end_}'
            ),
            features AS (
                SELECT
                    /*+ BROADCAST(target) */
                    target.report_dt,
                    target.epk_id,
                    {querry2[:-1]}

                FROM
                    target
                    LEFT JOIN agg agg_3m USING (epk_id)
                    LEFT JOIN agg agg_6m USING (epk_id)
                    LEFT JOIN agg agg_9m USING (epk_id)
                    LEFT JOIN agg agg_12m USING (epk_id)
                WHERE
                    agg_3m.report_dt BETWEEN target.report_dt_3m AND target.report_dt
                    AND agg_6m.report_dt BETWEEN target.report_dt_6m AND target.report_dt
                    AND agg_9m.report_dt BETWEEN target.report_dt_9m AND target.report_dt
                    AND agg_12m.report_dt BETWEEN target.report_dt_12m AND target.report_dt
                GROUP BY
                    target.report_dt,
                    target.epk_id
            )

            SELECT * FROM features
            ''')

            if oot_month is None:
                if not check_table_exist(output_scheme, f'{model_name}_aggr_dynamic_features_{part_table}_train'):
                    pos_dynamic_features.write.saveAsTable(f"{output_scheme}.{model_name}_aggr_dynamic_features_{part_table}_train", mode = 'overwrite')
            else:
                if not check_table_exist(output_scheme, f'{model_name}_aggr_dynamic_features_{part_table}_train'):
                    pos_dynamic_features.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_aggr_dynamic_features_{part_table}_train", mode = 'overwrite')
                if not check_table_exist(output_scheme, f'{model_name}_aggr_dynamic_features_{part_table}_oot'):
                    pos_dynamic_features.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_aggr_dynamic_features_{part_table}_oot", mode = 'overwrite')
            print(f'aggr_dynamic_features_{part_table}: done')
    
    def embeddings_fl():
        embeddings_fl = (spark.read.table(tables_dict['embeddings_fl'])  
                                   .withColumn('report_dt',  F.last_day(F.add_months(F.to_date('report_dt'), -1)))                      
        )

        embeddings_fl = (embeddings_fl.where(F.col('report_dt').isin(dates)).withColumn('report_dt', F.to_date('report_dt'))
                                              .join(target.select('epk_id', 'report_dt'),
                                                    how = 'inner',
                                                    on = ['epk_id', 'report_dt'],
                                                    )
              )
        if oot_month is None:
            if not check_table_exist(output_scheme, f'{model_name}_embeddings_fl_features_train'):
                embeddings_fl.write.saveAsTable(f"{output_scheme}.{model_name}_embeddings_fl_features_train", mode = 'overwrite')
        else:
            if not check_table_exist(output_scheme, f'{model_name}_embeddings_fl_features_train'):
                embeddings_fl.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_embeddings_fl_features_train", mode = 'overwrite')
            if not check_table_exist(output_scheme, f'{model_name}_embeddings_fl_features_oot'):
                embeddings_fl.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_embeddings_fl_features_oot", mode = 'overwrite')
        print('embeddings_fl_features: done')
        
    def pos_embeddings_fl():
        pos_embeddings_fl = (spark.read.table(tables_dict['pos_embeddings_fl'])  
                                .where(F.last_day(F.col('mon')).isin(dates))
                                .withColumn('report_dt', F.to_date('mon'))
        )

        pos_embeddings_fl = (pos_embeddings_fl.withColumn('report_dt', F.to_date('report_dt'))
                                              .join(target.select('epk_id', 'report_dt'),
                                                    how = 'inner',
                                                    on = ['epk_id', 'report_dt'],
                                                    )
              )
        if oot_month is None:
            if not check_table_exist(output_scheme, f'{model_name}_pos_embeddings_fl_features_train'):
                pos_embeddings_fl.write.saveAsTable(f"{output_scheme}.{model_name}_pos_embeddings_fl_features_train", mode = 'overwrite')
        else:
            if not check_table_exist(output_scheme, f'{model_name}_pos_embeddings_fl_features_train'):
                pos_embeddings_fl.filter(F.col('report_dt') != oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_pos_embeddings_fl_features_train", mode = 'overwrite')
            if not check_table_exist(output_scheme, f'{model_name}_pos_embeddings_fl_features_oot'):
                pos_embeddings_fl.filter(F.col('report_dt') == oot_month).write.saveAsTable(f"{output_scheme}.{model_name}_pos_embeddings_fl_features_oot", mode = 'overwrite')
        print('pos_embeddings_fl_features: done')
        

    tables = ['_agg_features_train','_arrests_features_train','_feedbacks_features_train','_card_transactions_10d_features_train',
              '_aggr_dynamic_features_1_train', '_aggr_dynamic_features_2_train', '_aggr_dynamic_features_3_train',
              '_vsp_visits_features_train','_card_transactions_features_train',
              '_flows12_features_train','_pos_dynamic_features_train', '_embeddings_fl_features_train', '_pos_embeddings_fl_features_train'
              '_aggr_dynamic_features_1_oot', '_aggr_dynamic_features_2_oot', '_aggr_dynamic_features_3_oot', '_card_transactions_10d_features_oot',
              '_agg_features_oot','_arrests_features_oot','_feedbacks_features_oot', 
              '_vsp_visits_features_oot','_card_transactions_features_oot',
              '_flows12_features_oot','_pos_dynamic_features_oot', '_embeddings_fl_features_oot', '_pos_embeddings_fl_features_oot']
    
    if mode == 'overwrite':
        for t in tables:
            _terminal_ld_3_0_bash_run(f"""hdfs dfs -rm -r -skipTrash 'hdfs://arnsdpsbx/user/team/team_ss/{model_name}{t}'""")
            try:
                spark.sql(f"""drop table {output_scheme}.{model_name}{t}""")
            except:
                pass

    
    agg_features()
    arrests_features()
    feedbacks_features()
    #vsp_visits_features()
    card_transactions_features() 
    card_transactions_10d_features()
    flows12_features()
    pos_dynamic_features()
    embeddings_fl()
    pos_embeddings_fl()
    aggr_dynamic_features()
    
    
    agg_features = spark.read.table(f"{output_scheme}.{model_name}_agg_features_train")
    arrests_features = spark.read.table(f"{output_scheme}.{model_name}_arrests_features_train")
    feedbacks_features = spark.read.table(f"{output_scheme}.{model_name}_feedbacks_features_train")
    #vsp_visits_features = spark.read.table(f"{output_scheme}.{model_name}_vsp_visits_features_train")
    card_transactions_features = spark.read.table(f"{output_scheme}.{model_name}_card_transactions_features_train")
    card_transactions_10d_features = spark.read.table(f"{output_scheme}.{model_name}_card_transactions_10d_features_train")
    flows12_features = spark.read.table(f"{output_scheme}.{model_name}_flows12_features_train")
    pos_dynamic_features = spark.read.table(f"{output_scheme}.{model_name}_pos_dynamic_features_train")
    embeddings_fl_features = spark.read.table(f"{output_scheme}.{model_name}_embeddings_fl_features_train")
    pos_embeddings_fl_features = spark.read.table(f"{output_scheme}.{model_name}_pos_embeddings_fl_features_train")
    aggr_dynamic_features_1 = spark.read.table(f"{output_scheme}.{model_name}_aggr_dynamic_features_1_train")
    aggr_dynamic_features_2 = spark.read.table(f"{output_scheme}.{model_name}_aggr_dynamic_features_2_train")
    aggr_dynamic_features_3 = spark.read.table(f"{output_scheme}.{model_name}_aggr_dynamic_features_3_train")
    
    if oot_month is not None:
        target_train = target.filter(F.col('report_dt') != oot_month)
        target_oot = target.filter(F.col('report_dt') == oot_month)
    else:
        target_train = target
        
    all_features_train = (F.broadcast(target_train)
                                .join(agg_features.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                .join(arrests_features.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                .join(feedbacks_features.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                #.join(vsp_visits_features, how = 'left', on = ['epk_id','report_dt'])
                                .join(card_transactions_features.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                .join(card_transactions_10d_features.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                .join(flows12_features.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                .join(pos_dynamic_features.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                .join(embeddings_fl_features.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                .join(pos_embeddings_fl_features.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                .join(aggr_dynamic_features_1.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                .join(aggr_dynamic_features_2.distinct(), how = 'left', on = ['epk_id','report_dt'])
                                .join(aggr_dynamic_features_3.distinct(), how = 'left', on = ['epk_id','report_dt'])
                        ).repartition(200) 
    all_features_train.write.saveAsTable(f"{output_scheme}.{model_name}_all_features_train", mode = 'overwrite')
    print('all_features_train: done')
    print(f"{output_scheme}.{model_name}_all_features_train")
    train = spark.read.table(f"{output_scheme}.{model_name}_all_features_train")
    fix_spark_types(train).write.saveAsTable(f"{output_scheme}.{model_name}_all_features_fix_types_train", mode = 'overwrite')
    #train_fix = spark.read.table(f"{output_scheme}.{model_name}_all_features_fix_types_train")
    #train_filtred = remove_correlated_features(train_fix, threshold = 0.99)
    #train_filtred.write.saveAsTable(f"{output_scheme}.{model_name}_all_features_fix_types_filtred_train", mode = 'overwrite')
    print('all_features_fix: done')
    print(f"{output_scheme}.{model_name}_all_features_fix_types_train: done")
    
    
    
    if oot_month is not None:
        agg_features = spark.read.table(f"{output_scheme}.{model_name}_agg_features_oot")
        feedbacks_features = spark.read.table(f"{output_scheme}.{model_name}_feedbacks_features_oot")
        arrests_features = spark.read.table(f"{output_scheme}.{model_name}_arrests_features_oot")
        ##vsp_visits_features = spark.read.table(f"{output_scheme}.{model_name}_vsp_visits_features_oot")
        card_transactions_features = spark.read.table(f"{output_scheme}.{model_name}_card_transactions_features_oot")
        card_transactions_10d_features = spark.read.table(f"{output_scheme}.{model_name}_card_transactions_10d_features_oot")
        flows12_features = spark.read.table(f"{output_scheme}.{model_name}_flows12_features_oot")
        pos_dynamic_features = spark.read.table(f"{output_scheme}.{model_name}_pos_dynamic_features_oot")
        embeddings_fl_features = spark.read.table(f"{output_scheme}.{model_name}_embeddings_fl_features_oot")
        pos_embeddings_fl_features = spark.read.table(f"{output_scheme}.{model_name}_pos_embeddings_fl_features_oot")
        aggr_dynamic_features_1 = spark.read.table(f"{output_scheme}.{model_name}_aggr_dynamic_features_1_oot")
        aggr_dynamic_features_2 = spark.read.table(f"{output_scheme}.{model_name}_aggr_dynamic_features_2_oot")
        aggr_dynamic_features_3 = spark.read.table(f"{output_scheme}.{model_name}_aggr_dynamic_features_3_oot")
        
        all_features_oot = (target_oot
                                   .join(agg_features, how = 'left', on = ['epk_id','report_dt'])
                                   .join(arrests_features, how = 'left', on = ['epk_id','report_dt'])
                                   .join(feedbacks_features, how = 'left', on = ['epk_id','report_dt'])
                                   #.join(vsp_visits_features, how = 'left', on = ['epk_id','report_dt'])
                                   .join(card_transactions_features, how = 'left', on = ['epk_id','report_dt'])
                                   .join(card_transactions_10d_features, how = 'left', on = ['epk_id','report_dt'])
                                   .join(flows12_features, how = 'left', on = ['epk_id','report_dt'])
                                   .join(pos_dynamic_features, how = 'left', on = ['epk_id','report_dt'])
                                   .join(embeddings_fl_features, how = 'left', on = ['epk_id','report_dt'])
                                   .join(pos_embeddings_fl_features, how = 'left', on = ['epk_id','report_dt'])
                                   .join(aggr_dynamic_features_1, how = 'left', on = ['epk_id','report_dt'])
                                   .join(aggr_dynamic_features_2, how = 'left', on = ['epk_id','report_dt'])
                                   .join(aggr_dynamic_features_3, how = 'left', on = ['epk_id','report_dt'])
                            ).distinct().repartition(200)

        all_features_oot.write.saveAsTable(f"{output_scheme}.{model_name}_all_features_oot", mode = 'overwrite')
        print('all_features_oot: done')
        print(f"{output_scheme}.{model_name}_all_features_oot")
        oot = spark.read.table(f"{output_scheme}.{model_name}_all_features_oot")
        fix_spark_types(oot).write.saveAsTable(f"{output_scheme}.{model_name}_all_features_fix_types_oot", mode = 'overwrite')
        #train_fix = spark.read.table(f"{output_scheme}.{model_name}_all_features_fix_types_train")
        #train_filtred = remove_correlated_features(train_fix, threshold = 0.99)
        #train_filtred.write.saveAsTable(f"{output_scheme}.{model_name}_all_features_fix_types_filtred_train", mode = 'overwrite')
        print('all_features_fix: done')
        print(f"{output_scheme}.{model_name}_all_features_fix_types_oot: done")

        
