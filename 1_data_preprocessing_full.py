import os, sys, re, csv
import def_function as func
from optparse import OptionParser
from datetime import datetime as dt
from datetime import timedelta
import pickle as pkl
import statistics
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import random

## (1) get case/control dataset 
def get_triple_ptSK_set(new_outcome_input_path):
    triple_ptSK_set = set()
    with open(new_outcome_input_path, 'r') as outcome_file: 
        for row in csv.reader(outcome_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            'day_gap-8', 'DAPT_med_label_set-28'
            if '3' in row[28] and int(row[8]) >= 0:                
                triple_ptSK_set.add(row[1])
        print(f'len of triple_ptSK_set is {len(triple_ptSK_set)}')
        # len(with_anticoagu_set) = 6814
    return triple_ptSK_set

def get_new_diag_id_code_dic(diag_head_input_path):
    diag_id_code_dic = {}
    code_set = set()
    with open(diag_head_input_path, 'r') as diag_head_file:
        # func.print_headrow_txt(diag_head_file)
        # ['diagnosis_id-0', 'diagnosis_type-1', 'diagnosis_code-2', 'diagnosis_description-3']
        for data in diag_head_file:
            row = data.strip().split('\t')
            # print(row)
            if row[0].isdigit():
                code_set.add(row[2])
                if row[1] == 'ICD9':
                    new_code = 'D_9_' + row[2].replace('.', '')
                elif row[1] == 'ICD10-CM' or row[1] == 'ICD10_CA':
                    new_code = 'D_10_' + row[2].replace('.', '')
                diag_id_code_dic[row[0]] = new_code
            # ICD9: D_9_71610
            # ICD10-CM: 210569 D_10_S62590
            # len(code_set) = 116661
            # total row : 210728; a lot of codes share the same form
        print('Done - get_new_diag_id_code_dic')
    return diag_id_code_dic

def get_new_event_id_group_name_dic(event_head_input_path):
    event_id_group_name_dic = {}
    with open(event_head_input_path, 'r') as event_head_file:
        # func.print_headrow_txt(event_head_file)
        # ['event_code_id-0', 'event_code_desc-1', 'event_code_display-2', 'event_code_group-3', 'event_code_category-4']
        # ['7-0', 'Allergies, Food Allergy Screen-1', 'Allergy Food-2', 'Allergies-3', 'Allergy-4']
        for data in event_head_file:
            row = data.strip().split('\t')
            if row[0].isdigit():
                new_code_name = 'E_' + row[3].lower().replace('.', '') 
                event_id_group_name_dic[row[0]] = new_code_name
                # print(row[0], new_code_name)
        print('Done - get_new_event_id_group_name_dic')
    return event_id_group_name_dic

def get_new_event_id_code_display_dic(event_head_input_path):
    event_id_code_display_dic = {}
    with open(event_head_input_path, 'r') as event_head_file:
        # ['event_code_id-0', 'event_code_desc-1', 'event_code_display-2', 'event_code_group-3', 'event_code_category-4']
        # ['7-0', 'Allergies, Food Allergy Screen-1', 'Allergy Food-2', 'Allergies-3', 'Allergy-4']
        for data in event_head_file:
            row = data.strip().split('\t')
            if row[0].isdigit():
                new_code_name = 'E_' + row[2].lower().replace('.', '')
                event_id_code_display_dic[row[0]] = new_code_name
                # 7 E_allergy food
                # 37 E_bk pn l4
                # 44 E_bk pn s3
        print('Done - get_new_event_id_code_display_dic')

    return event_id_code_display_dic

def get_new_lab_id_code_dic(lab_head_input_path):
    lab_id_code_dic = {}
    with open(lab_head_input_path, 'r') as lab_head_file:       
        # func.print_headrow_txt(lab_head_file)
        # ['lab_procedure_id-0', 'lab_procedure_mnemonic-1', 'lab_procedure_name-2', 'lab_procedure_group-3', 'lab_super_group-4', 'loinc_code-5', 'loinc_ind-6', 'loinc_long-7', 'loinc_short-8']
        for data in lab_head_file:
            row = data.strip().split('\t')
            if row[0].isdigit():
            # 'lab_procedure_name-2'
                new_lab_procedure_name = 'L_' + row[2].lower().replace('.', '') 
                lab_id_code_dic[row[0]] = new_lab_procedure_name
        ## 'loinc_code-5': 301 code loinc_code == 'NULL'
        ## len(lab_id_code_dic) == 23520
        print('Done - get_new_lab_id_code_dic')
    
    return lab_id_code_dic

def get_lab_with_result_code(row):    
    # L_lonincCode_categorical - get specific code from row
    # combine lab results and normal high/low value
    # ['patient_sk-0', 'encounter_id-1', 'detail_lab_procedure_id-2', 'lab_procedure_name-3', 'lab_procedure_group-4', 'loinc_code-5', 'reporting_priority_id-6', 'lab_result_type_id-7', 'lab_result_type_desc-8', 'lab_completed_dt_tm-9', 'lab_performed_dt_tm-10', 'numeric_result-11', 'result_units_id-12', 'normal_range_high-13', 'normal_range_low-14', 'merged_encid_label-15']
    ## (Counter(row_len_list)): Counter({14: 8399919, 16: 7629578, 15: 139165}) 
    
    # len(row) == 14: ['109068381', '572327572', '47', 'Sodium, Serum', 'Sodium Test', '2951-2', '11', '7', 'Numeric', '2016-08-13 00:53:22', '2016-08-13 00:46:29', '143'-11result, '203'-12-unit, '0'-13]
    # len(row) == 15: ['107967826', '303607545', '903', 'Cholesterol Screen', 'Cholesterol Test', '2093-3', '7', '12', 'NULL', '2013-12-19 21:43:00', '2013-12-19 21:43:00', '162'-11result, '179'-12-unit, '<200'-13, '0'-14]
    # len(row) == 16: ['107917674', '327779226', '62', 'Protein Total, Serum', 'Protein Test', '2885-2', '7', '12', 'NULL', '2014-04-28 16:42:00', '2014-04-28 16:41:00', '6.6'-11result, '50'-12id, '8'-13high, '6'-14low, '0'-15]
            
    # get code: 'detail_lab_procedure_id-2', 'loinc_code-5'; cate is value category: L-low, H-high, N-normal 
    new_lab_code = ''
    cate = ''
    if row[5] and row[5] != 'NULL':
        code = row[5]
    else: 
        code = row[2]

    # #  only len(row) == 16 has low and high value 
    # # get categorical value: 'numeric_result-11', 'normal_range_high-13', 'normal_range_low-14'
    if len(row) == 16 and func.is_number(row[11]) and func.is_number(row[13]) and func.is_number(row[14]):
        result, high_value, low_value = float(row[11]), float(row[13]), float(row[14])
        if result < high_value and result < low_value:
            cate = 'L'
        elif result > high_value and result > low_value:
            cate = 'H'
        elif (result > high_value and result > low_value) or (result < high_value and result > low_value):
            cate = 'N'
        if cate:
            new_lab_code = 'L_' + code + '_' + cate
        else: 
            new_lab_code = 'L_' + code
    else: 
        new_lab_code = 'L_' + code

    return new_lab_code     
                    
def get_new_med_id_generic_name_dic(med_head_input_path):
    med_id_generic_name_dic = {}
    with open(med_head_input_path, 'r') as med_head_file:       
        # func.print_headrow_txt(med_head_file)
        # ['medication_id-0', 'ndc_code-1', 'brand_name-2', 'generic_name-3', 'product_strength_description-4', 'route_description-5', 'dose_form_description-6', 'obsolete_dt_tm-7']
        for data in med_head_file:
            row = data.strip().split('\t')
            if row[0].isdigit():
                new_generic_name = 'M_' + row[3].lower().replace('.', '')                
                med_id_generic_name_dic[row[0]] = new_generic_name
        ## 8101307 M_ertugliflozin-sitagliptin    
        print('Done - get_new_med_id_generic_name_dic') 
    return med_id_generic_name_dic

def get_new_proce_id_code_dic(proce_head_input_path):
    proce_id_code_dic = {}
    with open(proce_head_input_path, 'r') as proce_head_file:       
        # func.print_headrow_txt(proce_head_file)
        # ['procedure_id-0', 'procedure_type-1', 'procedure_code-2', 'procedure_description-3']
        procedure_type_set = set()
        ## procedure_type_set = {'CPT4', 'ICD10-PCS', 'ICD9', 'HCPCS'}
        for data in proce_head_file:
            row = data.strip().split('\t')
            if row[0].isdigit():
                procedure_type_set.add(row[1])
                if row[1] == 'HCPCS':
                    new_code = 'P_' + 'H_' + row[2].replace('.', '')
                elif row[1] == 'CPT4':
                    new_code = 'P_' + 'C_' + row[2].replace('.', '')
                else: 
                    new_code = 'P_' + 'I_' + row[2].replace('.', '')
                proce_id_code_dic[row[0]] = new_code

        # 170701 P_I_047N441
        ## len(proce_id_code_dic) = 102366
        print('Done - get_new_proce_id_code_dic') 
   
    return proce_id_code_dic

def get_qualified_ptSK_set_simplified(new_outcome_input_path, triple_DAPT_ptSK_set, input_stop_daygap, conti_duration_daygap, prediction_end_daygap):
    old_ptSK = ''
    # DAPT_stop_daygap_list = []
    all_outcome_lists = []
    all_enc_list = []                        
    DAPT_positive_daygap_list = []
    DAPT_enc_list = []
    DAPT_info_list = []
    qualified_ptSK_set = set()
    # ptSK_stop_enc_dic = {}  
    # key is enc_id with daygap <= 0; value is discharged date time 
    before_stop_enc_daygap_dic = {}
    # key is ptSK; value is before_stop_enc_daygap_dic
    qualified_ptSK_before_index_enc_dic = {}
    # to append input encounter length 
    input_enc_len_list = []
    count = 0
    count_daygap = 0
    prediction_end_daygap = 730
    with open(new_outcome_input_path, 'r') as new_outcome_file:
        # ['-0', 'patient_sk-1', 'encounter_id-2', 'age_in_years-3', 'age_group_label-4', 'admitted_dt_tm-5', 'discharged_dt_tm-6', 'index_encounter_label-7', 'day_gap-8', 'final_bleeding_label-9', 'bleeding_code_set-10', 'final_traumatic_bleeding_label-11', 'traumatic_bleeding_code_set-12', 'final_GI_bleeding_label-13', 'GI_bleeding_code_set-14', 'final_transfusion_label-15', 'transfusion_code_set-16', 'final_ischemic_label-17', 'ischemic_code_set-18', 'final_acute_ischemic_event_label-19', 'acute_ischemic_code_set-20', 'final_eluting_stent_label-21', 'eluting_stent_code_set-22', 'final_revascularization_label-23', 'revascularization_code_set-24', 'final_stroke_label-25', 'stroke_code_set-26', 'DAPT_final_label-27', 'DAPT_med_label_set-28', 'DAPT_med_generic_name_set-29', 'expire_label-30', 'stroke_ischemic_label-31', 'stroke_ischemic_code_set_new-32', 'stroke_bleeding_label-33', 'stroke_bleeding_code_set_new-34']
        # ### get the first part of DAPT time gap < 180 days
        for row in csv.reader(new_outcome_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            if row[1].isdigit():
            # if row[1].isdigit() and row[1] not in triple_DAPT_ptSK_set:          
                cur_ptSK = row[1]
                if old_ptSK == '' or old_ptSK == cur_ptSK:
                    daygap = int(row[8])
                    discharged_dt = row[6].split(' ')[0] # only needs date, no hour-minute-sec
                    # record encounter with daygap <= 0
                    # 'encounter_id-2', 'discharged_dt_tm-6'
                    # input_stop_daygap: if input stopped at index encounter, input_stop_daygap = 0; 
                    if daygap <= input_stop_daygap and daygap >= -730:
                        if row[2] not in before_stop_enc_daygap_dic:
                            if count_daygap < 15:
                                print(count_daygap+1, discharged_dt)
                                count_daygap += 1
                            before_stop_enc_daygap_dic[row[2]] = discharged_dt
                    if daygap >= 0:
                        # record all lists and all encounter id after index
                        all_outcome_lists.append(row)
                        all_enc_list.append(row[2])                        
                        if row[27] != '0':
                            # 'DAPT_final_label-27', record DAPT related medication daygap and encounter id
                            DAPT_positive_daygap_list.append(daygap)
                            DAPT_enc_list.append(row[2])    
                            DAPT_info_list.append(row[27])                   
                else:
                    ## if input encounter number more than 100, get the last 100 (closer to index encounter) 
                    if len(before_stop_enc_daygap_dic) > 100:
                        before_stop_enc_daygap_dic = dict(list(before_stop_enc_daygap_dic.items())[-100:]) 

                    if prediction_end_daygap > conti_duration_daygap:
                        first_label = 0
                        second_label = 0
                        first_daygap_list = []
                        second_daygap_list = []
                        for _, pos_daygap in enumerate(DAPT_positive_daygap_list):                       
                            if pos_daygap > 0 and pos_daygap < conti_duration_daygap:
                                first_label = 1
                                if first_label:
                                    first_daygap_list.append(pos_daygap)
                            elif pos_daygap > conti_duration_daygap and pos_daygap < prediction_end_daygap:
                                second_label = 1
                                if second_label:
                                    second_daygap_list.append(pos_daygap)
                            
                            if first_label and second_label:                                
                                # if second_daygap_list[0] - first_daygap_list[-1] < conti_duration_daygap:
                                qualified_ptSK_set.add(old_ptSK)
                                if count < 5:
                                    print(DAPT_positive_daygap_list)
                                    print(first_daygap_list, second_daygap_list, (second_daygap_list[0] - first_daygap_list[-1]))
                                    print(DAPT_info_list, '\n')
                                    count += 1

                                qualified_ptSK_before_index_enc_dic[old_ptSK] = before_stop_enc_daygap_dic
                                input_enc_len_list.append(len(before_stop_enc_daygap_dic))
                                break
                            
                    elif prediction_end_daygap <= conti_duration_daygap:
                        for _, pos_daygap in enumerate(DAPT_positive_daygap_list):
                            if pos_daygap > 0 and pos_daygap < conti_duration_daygap:
                                qualified_ptSK_set.add(old_ptSK)
                                if count < 5:
                                    print(DAPT_positive_daygap_list)
                                    print(DAPT_info_list, '\n')
                                    count += 1

                                qualified_ptSK_before_index_enc_dic[old_ptSK] = before_stop_enc_daygap_dic
                                # append input encounter length
                                
                                break

                    # to empty the set element
                    all_outcome_lists = []
                    all_enc_list = []                        
                    DAPT_positive_daygap_list = []
                    DAPT_enc_list = []
                    DAPT_info_list = []
                    before_stop_enc_daygap_dic = {}
                    first_label = 0
                    second_label = 0
                    first_daygap_list = []
                    second_daygap_list = []

                    # to write the current new row
                    daygap = int(row[8])
                    discharged_dt = row[6].split(' ')[0]
                    if daygap <= input_stop_daygap and daygap >= -730:
                        if row[2] not in before_stop_enc_daygap_dic:
                            before_stop_enc_daygap_dic[row[2]] = discharged_dt
                    if daygap >= 0:
                        # record all lists and all encounter id after index
                        all_outcome_lists.append(row)
                        all_enc_list.append(row[2])                        
                        if row[27] != '0':
                            # record DAPT related medication daygap and encounter id
                            DAPT_positive_daygap_list.append(daygap)
                            DAPT_enc_list.append(row[2])   

                old_ptSK = cur_ptSK
        
        # to write the last patient
        if len(before_stop_enc_daygap_dic) > 100:
            before_stop_enc_daygap_dic = dict(list(before_stop_enc_daygap_dic.items())[-100:]) 
        if prediction_end_daygap > conti_duration_daygap:
            first_label = 0
            second_label = 0
            first_daygap_list = []
            second_daygap_list = []
            for _, pos_daygap in enumerate(DAPT_positive_daygap_list):                       
                if pos_daygap > 0 and pos_daygap < conti_duration_daygap:
                    first_label = 1
                    if first_label:
                        first_daygap_list.append(pos_daygap)
                elif pos_daygap > conti_duration_daygap and pos_daygap < prediction_end_daygap:
                    second_label = 1
                    if second_label:
                        second_daygap_list.append(pos_daygap)
                
                if first_label and second_label:                                
                    # if second_daygap_list[0] - first_daygap_list[-1] < conti_duration_daygap:
                    qualified_ptSK_set.add(old_ptSK)
                    print(DAPT_positive_daygap_list)
                    print(first_daygap_list, second_daygap_list, (second_daygap_list[0] - first_daygap_list[-1]), '\n')
                    qualified_ptSK_before_index_enc_dic[old_ptSK] = before_stop_enc_daygap_dic
                    input_enc_len_list.append(len(before_stop_enc_daygap_dic))
                    break
                
        elif prediction_end_daygap <= conti_duration_daygap:
            for _, pos_daygap in enumerate(DAPT_positive_daygap_list):
                if pos_daygap > 0 and pos_daygap < conti_duration_daygap:
                    qualified_ptSK_set.add(old_ptSK)
                    print(DAPT_positive_daygap_list)
                    qualified_ptSK_before_index_enc_dic[old_ptSK] = before_stop_enc_daygap_dic
                    # input_enc_len_list.append(len(before_stop_enc_daygap_dic))
                    break
        
        print(Counter(sorted(input_enc_len_list)))
        print(sorted(set(input_enc_len_list)))
        print(f'max is {max(input_enc_len_list)}, min is {min(input_enc_len_list)}, average is {statistics.mean(input_enc_len_list)}')

        print(f'conti_duration_daygap is {conti_duration_daygap}, input_stop_daygap is {input_stop_daygap}, prediction_end_daygap is {prediction_end_daygap}, len of qualified_ptSK_set is {len(qualified_ptSK_set)}')

        print('Done - get_qualified_ptSK_set')
        # simplified: 
        # conti_duration_daygap is 180 len of qualified_ptSK_set is 15316
        # conti_duration_daygap is 365 len of qualified_ptSK_set is 21864
    return qualified_ptSK_set, qualified_ptSK_before_index_enc_dic

def get_ischemic_ptSK_set_new(new_outcome_input_path, qualified_ptSK_set, prediction_start_daygap, prediction_end_daygap):
    """
    get case and control for ischemic event: >= 7d and <= 6m have ischemic, including ischemic stroke, case; if not, control
    new_outcome_file: include ischemic stroke and bleeding stroke
    """
    old_ptSK = ''
    ischemic_case_ptSK_set = set()
    ischemic_control_ptSK_set = set()
    ischemic_label_list = []
    with open(new_outcome_input_path, 'r') as new_outcome_file:
        for row in csv.reader(new_outcome_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            if row[1].isdigit() and row[1] in qualified_ptSK_set:
                cur_ptSK = row[1]
                if old_ptSK == '' or old_ptSK == cur_ptSK:
                    if cur_ptSK in qualified_ptSK_set:
                        daygap = int(row[8])
                        if daygap >= prediction_start_daygap and daygap <= prediction_end_daygap:
                            # 'final_acute_ischemic_event_label-19',  'final_eluting_stent_label-21', 'final_revascularization_label-23', 'stroke_ischemic_label-31'
                            if row[19] == '1' or row[21] == '1' or row[23] == '1':
                            #  or row[31] == '1':
                                ischemic_label_list.append('1')
                else: 
                    if ischemic_label_list:
                        ischemic_case_ptSK_set.add(old_ptSK)
                    else: 
                        ischemic_control_ptSK_set.add(old_ptSK)

                    # to empty the set element
                    ischemic_label_list = []

                    # to write the current new row
                    daygap = int(row[8])
                    if daygap >= prediction_start_daygap and daygap <= prediction_end_daygap:
                        if row[19] == '1' or row[21] == '1' or row[23] == '1':  
                        # or row[31] == '1':
                            ischemic_label_list.append('1')

                old_ptSK = cur_ptSK

        # write the last patient
        if ischemic_label_list:
            ischemic_case_ptSK_set.add(old_ptSK)
        else: 
            ischemic_control_ptSK_set.add(old_ptSK)
        # len(ischemic_case_ptSK_set) = 4279
        # len(ischemic_control_ptSK_set) = 5108
        print(f'len of new ischemic_case_ptSK_set is {len(ischemic_case_ptSK_set)}, len of new ischemic_control_ptSK_set is {len(ischemic_control_ptSK_set)}')
        print('Done - get_ischemic_ptSK_set_new')
        # len of new ischemic_case_ptSK_set is 4029, len of new ischemic_control_ptSK_set is 5358
    
    return ischemic_case_ptSK_set, ischemic_control_ptSK_set

def get_bleeding_ptSK_set_new(new_outcome_input_path, qualified_ptSK_set, prediction_start_daygap, prediction_end_daygap):
    ### get case and control for bleeding event: >= 7d and <= 6m have bleeding, or bleeding stroke, case; if not, control
    old_ptSK = '' 
    bleeding_case_ptSK_set = set()
    bleeding_control_ptSK_set = set()
    bleeding_label_list = []
    with open(new_outcome_input_path, 'r') as new_outcome_file:
        for row in csv.reader(new_outcome_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            if row[1].isdigit() and row[1] in qualified_ptSK_set:
                cur_ptSK = row[1]
                if old_ptSK == '' or old_ptSK == cur_ptSK:
                    daygap = int(row[8])
                    if daygap >= prediction_start_daygap and daygap <= prediction_end_daygap:
                        # 'final_bleeding_label-9', 'final_transfusion_label-15', 'stroke_bleeding_label-33',
                        if row[9] == '1' or row[15] == '1' or row[33] == '1':
                            bleeding_label_list.append('1')
                else: 
                    if bleeding_label_list:
                        bleeding_case_ptSK_set.add(old_ptSK)
                    else: 
                        bleeding_control_ptSK_set.add(old_ptSK)

                    # to empty the set element
                    bleeding_label_list = []

                    # to write the current new row
                    daygap = int(row[8])
                    if daygap >= prediction_start_daygap and daygap <= prediction_end_daygap:
                        if row[9] == '1' or row[15] == '1' or row[33] == '1':
                            bleeding_label_list.append('1')

                old_ptSK = cur_ptSK

        # write the last patient
        if bleeding_label_list:
            bleeding_case_ptSK_set.add(old_ptSK)
        else: 
            bleeding_control_ptSK_set.add(old_ptSK)
        # len(bleeding_case_ptSK_set) = 1212
        # len(bleeding_control_ptSK_set) = 8175 
        print(f'len of new bleeding_case_ptSK_set is {len(bleeding_case_ptSK_set)}, len of new bleeding_control_ptSK_set is {len(bleeding_control_ptSK_set)}')
        # len of new bleeding_case_ptSK_set is 1159, len of new bleeding_control_ptSK_set is 8228
        print('Done - new bleeding_case_ptSK_set')
        print('Done - new bleeding_control_ptSK_set')
            
    return bleeding_case_ptSK_set, bleeding_control_ptSK_set

def get_specific_file_input_dic(file_type, input_file_path, ptSK_set, qualified_ptSK_before_index_enc_dic, tm_label, code_dic):
    '''
    tm_label: == 1: event time; tm_label == 2: discharge time
    ''' 
    count = 0
    data_lists = []
    if file_type == 'age':
        with open(new_outcome_input_path, 'r') as new_outcome_file:
            # ['-0', 'patient_sk-1', 'encounter_id-2', 'age_in_years-3', 'age_group_label-4',  'discharged_dt_tm-6', ...]
            for row in csv.reader(new_outcome_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True): 
                if row[1] in ptSK_set:
                    before_index_enc_dic = qualified_ptSK_before_index_enc_dic[row[1]]
                    if row[2] in before_index_enc_dic:
                        # print(f'age is {row[1], row[3]}')
                        # age is ('213677036', 'nan')
                        age = int(float(row[3]))
                        age_code = 'A_' + str(age)
                        age_list = [row[1], age_code, before_index_enc_dic[row[2]]]
                        if count < 5:
                            print(age_list)
                            count += 1
                        data_lists.append(age_list)

    elif file_type == 'gender':       
        with open(history_input_path, 'r') as history_file:
            # ['-0', 'patient_sk-1', 'race-2', 'race_label-3', 'gender-4', 'gender_label-5',
            for row in csv.reader(history_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True): 
                if row[1] in ptSK_set:
                    before_index_enc_dic = qualified_ptSK_before_index_enc_dic[row[1]]
                    for _, discharged_dt in before_index_enc_dic.items():
                        ## gender_list: ['patient_sk-1', 'gender-4', 'discharged_dt]
                        gener_code = 'G_' + row[4]
                        gender_list = [row[1], gener_code, discharged_dt]
                        if count < 5:
                            print(gender_list)
                            count += 1
                        data_lists.append(gender_list)

    elif file_type == 'race':
        with open(history_input_path, 'r') as history_file:
            # ['-0', 'patient_sk-1', 'race-2', 'race_label-3', 'gender-4', 'gender_label-5',
            for row in csv.reader(history_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True): 
                if row[1] in ptSK_set:
                    before_index_enc_dic = qualified_ptSK_before_index_enc_dic[row[1]]
                    for _, discharged_dt in before_index_enc_dic.items():
                        ## race_list: ['patient_sk-1', 'race-2', 'discharged_dt] 
                        race_code = 'R_' + row[2]                       
                        race_list = [row[1], race_code, discharged_dt]
                        if count < 5:
                            print(race_list)
                            count += 1
                        data_lists.append(race_list)
      
    elif file_type == 'diagnosis':
        with open(orig_diag_input_path, 'r') as orig_diag_file: 
            #  ['patient_sk-0', 'encounter_id-1', 'diagnosis_id-2', 'diagnosis_priority-3', 'merged_encid_label-4']
            for row in csv.reader(orig_diag_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True): 
                if row[0].isdigit() and row[0] in ptSK_set:
                    # get index_enc_dic
                    before_index_enc_dic = qualified_ptSK_before_index_enc_dic[row[0]]
                    if row[1] in before_index_enc_dic:
                        diag_code = diag_id_code_dic[row[2]]
                        discharge_tm = before_index_enc_dic[row[1]]
                        diag_list = [row[0], diag_code, discharge_tm]
                        if count < 5:
                            print(diag_list)
                            count += 1
                        data_lists.append(diag_list)

    elif file_type == 'event':
        with open(orig_event_input_path, 'r') as orig_event_file:
            # ['patient_sk-0', 'encounter_id-1', 'lab_procedure_id-2', 'event_code_id-3', 'critical_high-4', 'critical_low-5', 'event_start_dt_tm-6', 'event_end_dt_tm-7', 'normal_high-8', 'normal_low-9', 'performed_dt_tm-10', 'result_value_num-11', 'merged_encid_label-12']
            for row in csv.reader(orig_event_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True): 
                if row[0].isdigit() and row[0] in ptSK_set:
                    # get index_enc_dic
                    before_index_enc_dic = qualified_ptSK_before_index_enc_dic[row[0]]
                    if row[1] in before_index_enc_dic:
                        discharge_tm = before_index_enc_dic[row[1]]
                        ## 'event_code_id-3', 
                        if row[3] and row[3].isdigit():
                            event_name = event_id_group_name_dic[row[3]]
                            if tm_label == 1:
                                ## keep original time that is not NULL
                                if row[6] and row[6] != 'NULL':
                                    new_discharge_tm = row[6].split(' ')[0]
                                elif row[7] and row[7] != 'NULL':
                                    new_discharge_tm = row[7].split(' ')[0]
                                else:
                                    new_discharge_tm = discharge_tm          
                            elif tm_label == 2:
                                new_discharge_tm = discharge_tm 
                            event_list = [row[0], event_name, new_discharge_tm]
                            if count < 5:
                                print(event_list)
                                count += 1
                            data_lists.append(event_list)

    elif file_type == 'medication':
        with open(orig_med_input_path, 'r') as orig_med_file:
            # ['patient_sk-0', 'encounter_id-1', 'medication_id-2', 'total_dispensed_doses-3', 'dose_quantity-4', 'initial_dose_quantity-5', 'order_strength-6', 'med_started_dt_tm-7', 'med_entered_dt_tm-8', 'med_stopped_dt_tm-9', 'med_discontinued_dt_tm-10', 'merged_encid_label-11']
            for row in csv.reader(orig_med_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True): 
                if row[0].isdigit() and row[0] in ptSK_set:
                    # get index_enc_dic
                    before_index_enc_dic = qualified_ptSK_before_index_enc_dic[row[0]]
                    if row[1] in before_index_enc_dic:
                        discharge_tm = before_index_enc_dic[row[1]]
                        generic_name_code = med_id_generic_name_dic[row[2]]
                        if tm_label == 1:
                            ## keep original time that is not NULL
                            if row[7] and row[7] != 'NULL':
                                new_discharge_tm = row[7].split(' ')[0]
                            elif row[8] and row[8] != 'NULL':
                                new_discharge_tm = row[8].split(' ')[0]
                            elif row[9] and row[9] != 'NULL':
                                new_discharge_tm = row[9].split(' ')[0]
                            else:
                                new_discharge_tm = discharge_tm    
                        elif tm_label == 2:
                            new_discharge_tm = discharge_tm    
                        med_list = [row[0], generic_name_code, new_discharge_tm]
                        if count < 5:
                            print(med_list)
                            count += 1
                        data_lists.append(med_list)

    elif file_type == 'lab':
        with open(orig_lab_input_path, 'r') as orig_lab_file:
            # ['patient_sk-0', 'encounter_id-1', 'detail_lab_procedure_id-2', 'lab_procedure_name-3', 'lab_procedure_group-4', 'loinc_code-5', 'reporting_priority_id-6', 'lab_result_type_id-7', 'lab_result_type_desc-8', 'lab_completed_dt_tm-9', 'lab_performed_dt_tm-10', 'numeric_result-11', 'result_units_id-12', 'normal_range_high-13', 'normal_range_low-14', 'merged_encid_label-15']
            for row in csv.reader(orig_lab_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True): 
                if row[0].isdigit() and row[0] in ptSK_set:
                    # get index_enc_dic
                    before_index_enc_dic = qualified_ptSK_before_index_enc_dic[row[0]]
                    if row[1] in before_index_enc_dic:
                        new_lab_code = get_lab_with_result_code(row)
                        discharge_tm = before_index_enc_dic[row[1]]
                        if tm_label == 1: # to use event time
                        ## keep original time that is not NULL
                            if row[10] and row[10] != 'NULL':
                                new_discharge_tm = row[10].split(' ')[0]
                            elif row[9] and row[9] != 'NULL':
                                new_discharge_tm = row[9].split(' ')[0]
                            else:
                                new_discharge_tm = discharge_tm    
                        elif tm_label == 2: # to use discharge tim
                            new_discharge_tm = discharge_tm 
                        lab_list = [row[0], new_lab_code, new_discharge_tm]
                        # if count < 5:
                        #     print(lab_list)
                        #     # print(row)
                        #     print(f'discharge_tm is {discharge_tm}, new_discharge_tm is {new_discharge_tm}')
                        #     count += 1
                        data_lists.append(lab_list)

    elif file_type == 'procedure':
        with open(orig_proce_input_path, 'r') as orig_proce_file:
            # ['patient_sk-0', 'encounter_id-1', 'procedure_id-2', 'procedure_code-3', 'procedure_type-4', 'procedure_cat-5', 'procedure_priority-6', 'procedure_dt_tm-7', 'merged_encid_label-8']   
            for row in csv.reader(orig_proce_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True): 
                if row[0].isdigit() and row[0] in ptSK_set:
                    # get index_enc_dic
                    before_index_enc_dic = qualified_ptSK_before_index_enc_dic[row[0]]
                    if row[1] in before_index_enc_dic:
                        new_proce_code = proce_id_code_dic[row[2]]
                        discharge_tm = before_index_enc_dic[row[1]]
                        if tm_label == 1:
                            ## keep original time that is not NULL
                            if row[7] and row[7] != 'NULL':
                                new_discharge_tm = row[7].split(' ')[0]
                            else:
                                new_discharge_tm = discharge_tm    
                        elif tm_label == 2:
                            new_discharge_tm = discharge_tm    
                        proce_list = [row[0], new_proce_code, new_discharge_tm]
                        # if count < 5:
                        #     print(proce_list)
                        #     # print(row)
                        #     print(f'discharge_tm is {discharge_tm}, new_discharge_tm is {new_discharge_tm}')
                        #     count += 1
                        data_lists.append(proce_list)
               
    return data_lists

def write_data_lists_to_tsv(all_all_data_lists, output_path):
    with open(output_path, 'wt') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        head_row = ['Pt_id', 'ICD', 'Time']
        writer.writerow(head_row)
        for lst in all_all_data_lists:
            writer.writerows(lst)

def deduplicate_tsv(output_path, dedup_output_path):
    # to depulicate the same row in tsv 
    with open(output_path, 'r') as output_file:
        df_data = pd.read_csv(output_file, sep = "\t")
        print(f'before depulication: row num is {len(df_data)}; column num is {len(df_data.columns)}') 
        df_data_result = df_data.drop_duplicates()
        print(f'after depulication: row num is {len(df_data_result)}; column num is {len(df_data_result.columns)}') 
        result_tsv = df_data_result.to_csv(dedup_output_path, sep = '\t', index = False, header = True)

    return result_tsv

## (2) splitting train/validation/test data
def get_input_file_1st_splitting(common_path, file_type):
    caseFile = common_path + 'dedup_' + file_type + '_case.tsv'
    controlFile = common_path + 'dedup_' + file_type + '_control.tsv' 
    typeFile = 'NA'
    # outFile = common_path + 'Results/' + file_type
    outFile = common_path + file_type
    cls_type = 'binary'
    pts_file_pre = 'NA'

    return caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre

def get_input_file_non_1st_splitting(common_path, first_splitting_folder_path, file_type):
    caseFile = common_path + 'dedup_' + file_type + '_case.tsv'
    controlFile = common_path + 'dedup_' + file_type + '_control.tsv' 
    typeFile = 'NA'
    outFile = common_path + file_type
    # outFile = common_path + 'Results/' + file_type
    cls_type = 'binary'
    pts_file_pre = first_splitting_folder_path + file_type + '.pts'
    # 2nd time: /data/fli2/Stent/Cerner/Results/For_ML/Results_20210219/Results_all_input_all_time_180_6m_window/Results/bleeding.pts.test

    return caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre

def spliting_train_test_data(caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre):
    #_start = timeit.timeit()
    # debug = False
    #np.random.seed(1)
    # time_list = []
    # dates_list =[]
    label_list = []
    pt_list = []

    print ("Loading cases and controls" ) 

    ## loading Case
    print('loading cases')
    data_case = pd.read_table(caseFile)
    # data_case.columns = ["Pt_id", "ICD", "Time", "tte"] # ?"tte": time to event
    # index: stenting   1st bleeding: 100 days; 150; 200
    # survival analysis: only get the first outcome event

    if cls_type=='surv':
        data_case = data_case[["Pt_id", "ICD", "Time", "tte"]]
    else:
        data_case = data_case[["Pt_id", "ICD", "Time"]]
    data_case['Label'] = 1
    #data_case=data_case[~(data_case["ICD"].str.startswith('P') |data_case["ICD"].str.startswith('L'))] ### use if you need to exclude or include only certain type of codes
    print('Case counts: ', data_case["Pt_id"].nunique())

    ## loading Control
    print('loading ctrls')
    data_control = pd.read_table(controlFile)
    if cls_type == 'surv':
        data_control = data_control[["Pt_id", "ICD", "Time", "tte"]]
    else:
        data_control = data_control[["Pt_id", "ICD", "Time"]]
    data_control['Label'] = 0

    #data_control=data_control[~(data_control["ICD"].str.startswith('P') | data_control["ICD"].str.startswith('L'))] ### use if you need to exclude certain type of codes
    print('Ctrl counts: ', data_control["Pt_id"].nunique())

    ### An example of sampling code: Control Sampling
    #print('ctrls sampling')       
    #ctr_sk=data_control["Pt_id"]
    #ctr_sk=ctr_sk.drop_duplicates()
    #ctr_sk_samp=ctr_sk.sample(n=samplesize_ctrl)
    #data_control=data_control[data_control["Pt_id"].isin(ctr_sk_samp.values.tolist())]

    data_l = pd.concat([data_case, data_control])
    print('total counts: ', data_l["Pt_id"].nunique())   

    ## loading the types
    if typeFile=='NA': 
       types = {"Zeropad":0}
    else:
      with open(typeFile, 'rb') as t2:
             types = pkl.load(t2)

    label_list = []
    pt_list = []
    dur_list = []
    newVisit_list = []
    count = 0

    for Pt, group in data_l.groupby('Pt_id'):
        data_i_c = [] # i - icd code
        data_dt_c = [] # dt - date
        for Time, subgroup in group.sort_values(['Time'], ascending = False).groupby('Time', sort=False): ### ascending=True normal order ascending=False reveresed order
            data_i_c.append(np.array(subgroup['ICD']).tolist())             
            data_dt_c.append(dt.strptime(Time, '%Y-%m-%d'))
        if len(data_i_c) > 0:
            # creating the duration in days between visits list, first visit marked with 0        
            v_dur_c = []
        if len(data_dt_c) <= 1:
            v_dur_c = [0]
        else:
            for jx in range(len(data_dt_c)):
                if jx == 0:
                    v_dur_c.append(jx)
                else:
                    #xx = (data_dt_c[jx]- data_dt_c[jx-1]).days ### normal order
                    xx = (data_dt_c[jx-1] - data_dt_c[jx]).days ## reversed order                            
                    v_dur_c.append(xx)                                  

        ### Diagnosis recoding
        newPatient_c = []
        for visit in data_i_c:
            newVisit_c = []
            for code in visit:
                if code in types: 
                    newVisit_c.append(types[code])
                else:                             
                    types[code] = max(types.values()) + 1 
                    # types[code] = len(types) + 1
                    newVisit_c.append(types[code])
            newPatient_c.append(newVisit_c)

        if len(data_i_c) > 0: ## only save non-empty entries
            if cls_type == 'surv':
                label_list.append([group.iloc[0]['Label'], group.iloc[0]['tte']]) #### LR ammended for surv
            else:
                label_list.append(group.iloc[0]['Label'])
            pt_list.append(Pt)
            newVisit_list.append(newPatient_c)
            dur_list.append(v_dur_c)

        count = count + 1
        if count % 1000 == 0: print ('processed %d pts' % count)

    ### Creating the full pickled lists ### uncomment if you need to dump the all data before splitting
    #pickle.dump(label_list, open(outFile+'.labels', 'wb'), -1)
    #pickle.dump(newVisit_list, open(outFile+'.visits', 'wb'), -1)
    pkl.dump(types, open(outFile + '.types', 'wb'), -1)
    #pickle.dump(pt_list, open(outFile+'.pts', 'wb'), -1)
    #pickle.dump(dur_list, open(outFile+'.days', 'wb'), -1)

    ### Random split to train, test and validation sets
    print("Splitting")

    if pts_file_pre == 'NA':
        print('random split')
        dataSize = len(pt_list)
        #np.random.seed(0)
        ind = np.random.permutation(dataSize)
        nTest = int(0.2 * dataSize)
        nValid = int(0.1 * dataSize)
        test_indices = ind[:nTest]
        valid_indices = ind[nTest:nTest+nValid]
        train_indices = ind[nTest+nValid:]
    else:
        print('loading previous splits')
        pt_train = pkl.load(open(pts_file_pre + '.train', 'rb'))
        pt_valid = pkl.load(open(pts_file_pre + '.valid', 'rb'))
        pt_test = pkl.load(open(pts_file_pre + '.test', 'rb'))
        test_indices = np.intersect1d(pt_list, pt_test, assume_unique=True, return_indices=True)[1]
        valid_indices= np.intersect1d(pt_list, pt_valid, assume_unique=True, return_indices=True)[1]
        train_indices= np.intersect1d(pt_list, pt_train, assume_unique=True, return_indices=True)[1]

    for subset in ['train', 'valid', 'test']:
        if subset =='train':
            indices = train_indices
        elif subset =='valid':
            indices = valid_indices
        elif subset =='test':
            indices = test_indices
        else: 
            print ('error')
            break
        
        #### below comments are mainly because I'm no longer need those theano RETAIN needed data, so comment for now
        #### only using Pts file , so keeping them for now
        
        #subset_x = [newVisit_list[i] for i in indices]
        #subset_y = [label_list[i] for i in indices]
        #subset_t = [dur_list[i] for i in indices]
        subset_p = [pt_list[i] for i in indices]
        #nseqfile = outFile +'.visits.'+subset
        #nlabfile = outFile +'.labels.'+subset
        #ntimefile = outFile +'.days.'+subset
        nptfile = outFile + '.pts.' + subset
        #pickle.dump(subset_x, open(nseqfile, 'wb'),protocol=2)
        #pickle.dump(subset_y, open(nlabfile, 'wb'),protocol=2)
        #pickle.dump(subset_t, open(ntimefile, 'wb'),protocol=2)
        pkl.dump(subset_p, open(nptfile, 'wb'), protocol=2)    
        
    ### Create the combined list for the Pytorch RNN
    fset = []
    print('Reparsing')
    for pt_idx in range(len(pt_list)):
        pt_sk = pt_list[pt_idx]
        pt_lbl = label_list[pt_idx]
        pt_vis = newVisit_list[pt_idx]
        pt_td = dur_list[pt_idx]
        n_seq = []
        for v in range(len(pt_vis)):
            nv = []
            nv.append([pt_td[v]])
            nv.append(pt_vis[v])                   
            n_seq.append(nv)
        n_pt = [pt_sk, pt_lbl, n_seq]
        fset.append(n_pt)              

    ### split the full combined set to the same as individual files
    train_set_full = [fset[i] for i in train_indices]
    test_set_full = [fset[i] for i in test_indices]
    valid_set_full = [fset[i] for i in valid_indices]
    ctrfilename = outFile + '.combined.train'
    ctstfilename = outFile + '.combined.test'
    cvalfilename = outFile + '.combined.valid'    
    pkl.dump(train_set_full, open(ctrfilename, 'wb'), -1)
    pkl.dump(test_set_full, open(ctstfilename, 'wb'), -1)
    pkl.dump(valid_set_full, open(cvalfilename, 'wb'), -1)
 
if __name__ == "__main__":
    # ../represents above level of the current folder
    head_input_common_path = func.get_common_path('../Data/Original_data/hf_d_data/')
    orig_input_common_path = func.get_common_path('../Data/Eluting_encid_merged_data/')    
    outcome_input_common_path = func.get_common_path('../../Qing_stent_class/Results/new_results_20201214/')
    dic_input_common_path = func.get_common_path('../Results/Dic/Disease_codes_dic/')    
    output_folder_name = 'Results_20210405_full_dischar_tm_seq_trun'
    output_part_common_path = func.get_common_path('../Results/For_ML/') + output_folder_name + '/'

    conti_duration_daygap = 365
    input_stop_daygap = 365
    prediction_start_daygap = 366
    prediction_end_daygap = 546
    
    # make empty folders to store results:
    # cd to the output_folder:
    # mkdir -p Results_all_input_dischar_tm_365_7d_180d_window
    # mkdir -p Results_all_input_dischar_tm_365_7d_365d_window
    # mkdir -p Results_all_input_dischar_tm_365_181d_365d_window
    # mkdir -p Results_all_input_dischar_tm_365_181d_270d_window
    # mkdir -p Results_all_input_dischar_tm_365_366d_546d_window
    # mkdir -p Results_all_input_dischar_tm_365_366d_730d_window
    
    # '''
    start_time = dt.now()
    print(f'start time to get multiple lists is {start_time}')
    log_name = output_folder_name[8:] + str(conti_duration_daygap) + '_' + str(prediction_start_daygap) + 'd_' + str(prediction_end_daygap) + 'd_preprocessing.log'    
    print(f'log name is {log_name}', '\n')   
    print(f'conti_duration_daygap is {conti_duration_daygap}d, input_stop_daygap is {input_stop_daygap}d, prediction_start_daygap is {prediction_start_daygap}d, prediction_end_daygap is {prediction_end_daygap}d')

    specific_folder_name = 'Results_all_input_dischar_tm_' + str(conti_duration_daygap) + '_' + str(prediction_start_daygap) + 'd_' + str(prediction_end_daygap) + 'd_window'

    # head file path   
    head_file_name_list = ['d_diagnosis.txt', 'd_event_code.txt', 'd_lab_procedure.txt', 'd_medication.txt', 'd_procedure.txt']
    head_input_path_list = []
    for head_file_name in head_file_name_list:
        head_input_path = head_input_common_path + head_file_name
        head_input_path_list.append(head_input_path)
    diag_head_input_path, event_head_input_path, lab_head_input_path, med_head_input_path, proce_head_input_path = head_input_path_list

    # orig data file path
    orig_file_name_list = ['eluting_diag_merged_same_disch.csv', 'eluting_event_merged_same_disch.csv', 'eluting_lab_merged_same_disch.csv', 'eluting_med_merged_same_disch.csv', 'eluting_proce_merged_same_disch.csv']
    orig_file_path_list = []
    for orig_file_name in orig_file_name_list:
        orig_file_path = orig_input_common_path + orig_file_name
        orig_file_path_list.append(orig_file_path)
    orig_diag_input_path, orig_event_input_path, orig_lab_input_path, orig_med_input_path, orig_proce_input_path = orig_file_path_list
  
    # outcome_input_path = outcome_input_common_path + 'filtered_outcome_encounter_level.csv'
    new_outcome_input_path = outcome_input_common_path + 'full_outcome_encounter_level_stroke_subgrouping_age18over.csv'
    history_input_path = outcome_input_common_path + 'full_history_patient_level_age18over.csv'

    # start processing, get id-code dic from head files
    to_get_code_dic_label = 1
    if to_get_code_dic_label:
        diag_id_code_dic = get_new_diag_id_code_dic(diag_head_input_path)
        event_id_group_name_dic = get_new_event_id_group_name_dic(event_head_input_path)
        event_id_code_display_dic = get_new_event_id_code_display_dic(event_head_input_path)
        lab_id_code_dic = get_new_lab_id_code_dic(lab_head_input_path)
        med_id_generic_name_dic = get_new_med_id_generic_name_dic(med_head_input_path)
        proce_id_code_dic = get_new_proce_id_code_dic(proce_head_input_path)
          
    # to get ptSK set that have triple DAPT therapy
    to_get_triple_DAPT_ptSK_set = 1
    if to_get_triple_DAPT_ptSK_set:
        triple_DAPT_ptSK_set = get_triple_ptSK_set(new_outcome_input_path)

    # get qualified ptSk sets 
    to_get_ptSK_set_label = 1
    if to_get_ptSK_set_label:
        ### get_qualified_ptSK_set_simplified(new_outcome_input_path, input_stop_daygap, conti_duration_daygap, prediction_end_daygap)
        qualified_ptSK_set, qualified_ptSK_before_index_enc_dic = get_qualified_ptSK_set_simplified(new_outcome_input_path, triple_DAPT_ptSK_set, input_stop_daygap, conti_duration_daygap, prediction_end_daygap)   

        ### get_ischemic_ptSK_set_new(new_outcome_input_path, qualified_ptSK_set, prediction_start_daygap, prediction_end_daygap)
        ischemic_case_ptSK_set, ischemic_control_ptSK_set = get_ischemic_ptSK_set_new(new_outcome_input_path, qualified_ptSK_set, prediction_start_daygap, prediction_end_daygap)

        # ### get_bleeding_ptSK_set_new(new_outcome_input_path, qualified_ptSK_set, prediction_start_daygap, prediction_end_daygap)
        bleeding_case_ptSK_set, bleeding_control_ptSK_set = get_bleeding_ptSK_set_new(new_outcome_input_path, qualified_ptSK_set, prediction_start_daygap, prediction_end_daygap)

    to_write_and_deduplicate_tsv_label = 1
    if to_write_and_deduplicate_tsv_label:
        ptSK_set_list = [ischemic_case_ptSK_set, ischemic_control_ptSK_set, bleeding_case_ptSK_set, bleeding_control_ptSK_set]
        output_file_name_list = ['ischemic_case.tsv', 'ischemic_control.tsv', 'bleeding_case.tsv', 'bleeding_control.tsv']
        for idx1, ptSK_set in enumerate(ptSK_set_list):
            ## tm_label = 1, all time (like med start time, procedure conducted time); tm_label = 2, discharged tm
            print(f'{idx1+1}. start generating {output_file_name_list[idx1]} for {specific_folder_name}')
            age_lists = get_specific_file_input_dic('age', new_outcome_input_path, ptSK_set, qualified_ptSK_before_index_enc_dic, tm_label = False, code_dic = False)

            gender_lists = get_specific_file_input_dic('gender', history_input_path, ptSK_set, qualified_ptSK_before_index_enc_dic, tm_label = False, code_dic = False)

            race_lists = get_specific_file_input_dic('race', history_input_path, ptSK_set, qualified_ptSK_before_index_enc_dic, tm_label = False, code_dic = False)

            diag_lists = get_specific_file_input_dic('diagnosis', orig_diag_input_path, ptSK_set, qualified_ptSK_before_index_enc_dic, 2, diag_id_code_dic)

            event_lists = get_specific_file_input_dic('event', orig_event_input_path, ptSK_set, qualified_ptSK_before_index_enc_dic, 2, event_id_group_name_dic)

            lab_lists = get_specific_file_input_dic('lab', orig_lab_input_path, ptSK_set, qualified_ptSK_before_index_enc_dic, 2, code_dic = False) 

            med_lists = get_specific_file_input_dic('medication', orig_med_input_path, ptSK_set, qualified_ptSK_before_index_enc_dic, 2, med_id_generic_name_dic)

            proce_lists = get_specific_file_input_dic('procedure', orig_proce_input_path, ptSK_set, qualified_ptSK_before_index_enc_dic, 2, proce_id_code_dic)
            
            all_input_disch_time_list = [age_lists, gender_lists, race_lists, diag_lists, event_lists, lab_lists, med_lists, proce_lists]   
            
            # for idx2, input_list in enumerate(all_input_lists):                    
            output_common_path = output_part_common_path + specific_folder_name + '/'
            output_path = output_common_path + output_file_name_list[idx1]
            dedup_output_path = output_common_path + 'dedup_' + output_file_name_list[idx1]   
            print(f'{idx1 + 1}. Start writing {output_file_name_list[idx1]}-{specific_folder_name}', '\n')            
            write_data_lists_to_tsv(all_input_disch_time_list, output_path)
            print(f'{idx1 + 1}. finish writing {output_file_name_list[idx1]}-{specific_folder_name}', '\n') 
            dedup_result_csv = deduplicate_tsv(output_path, dedup_output_path) 
            print(f'{idx1 + 1}. finish deduplicating {output_file_name_list[idx1]}-{specific_folder_name}', '\n')      
    
    end_time_case_ctl = dt.now()
    print(f'Time for get case/control is {end_time_case_ctl - start_time}', '\n')

    ## to split into train/valid/test files 
    to_split_train_test_label = 1
    if to_split_train_test_label:
        file_type_list = ['ischemic', 'bleeding']
        for idx_file_type1, file_type in enumerate(file_type_list):
            print(f'{idx_file_type1 + 1}. generating-{file_type} - tran/test/valid for {specific_folder_name}')
            
            split_input_file_path = output_part_common_path + specific_folder_name + '/'
            caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre = get_input_file_1st_splitting(split_input_file_path, file_type)
            
            spliting_train_test_data(caseFile, controlFile, typeFile, outFile, cls_type, pts_file_pre)
            print(f'{idx_file_type1 + 1}. finish generating - {file_type} - train/test/valid for {specific_folder_name}', '\n')
    
    end_time_train_test = dt.now()
    print(f'Time for get train/test/valid is {end_time_train_test - end_time_case_ctl}, total time is {end_time_train_test - start_time}')
