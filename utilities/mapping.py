

mapping_list = {
    "bourbon": {
        "shipping_date": "date",
        "item_number": "number",
        "number_of_cases": "number",
        "number_of_balls_in_case": "number",
        "number_of_pieces": "number",
        "quantity": "number",
        "unit": "",	
        "order_number": "",	
        '\t\nitem_number\n': "number",
        '\t\norder_number\n': "",
        'number_of_cases\n': "number",
        '\norder_number\n': "",
    },
    "sompo_holdings":{
        "client_name": "person_name",
        "date": "date",
        "department_name": "",	
        "copayment_ratio":	"number",
        "patient_status": "",	
        "hospital_name": "company_name",
        "receipt_amount": "number",
        #"receipt_stamp": "",
        "room_charge": "number",
        "total_non_insurance_burden": "number",
        "psychiatric_specialty_therapy": "number",
        "total_insurance_burden": "number",
        "insurance_covered_fee": "number",
        "total_insurance_burden_score": "number",
    },
    "myl":{
        "insurance_point_total": "number",
        "insurance_amount_total": "number",
        "patient_copayment_total": "number",
        "copayment_ratio": "number",
        "patient_status": "",	
        "billing_period": "date",
        "surgery_fee_score": "number",
        "surgery_fee_amount": "number",
        "radiation_score": "number",
        "radiation_amount":	"number",
        "advanced_medical_tech_amount": "number",
        "hospital_name": "company_name",
        'issued_date': "date",
    },
    "invoice":{
        "company_name":	"company_name",
        "branch_name": "",	
        "company_tel": "tel_fax",
        "company_fax": "tel_fax",
        "company_address": "address",
        "company_zipcode": "zipcode",
        "branch_address": "address",
        "branch_zipcode": "zipcode",
        "branch_tel": "tel_fax",
        "branch_fax": "tel_fax",
        "amount_excluding_tax": "number",
        "amount_tax": "number",
        "tax": "number",
        "amount_including_tax": "number",
        "bank_name": "",
        "bank_branch_name": "",	
        "account_type": "",
        "account_number": "number",
        "account_name": "",
        "item_name": "description",
        "item_quantity": "number",
        "item_quantity_item_unit": "number",
        "item_unit" : "",
        "item_unit_amount": "number",
        "item_total_excluding_tax": "number",
        "item_total_tax": "number",
        "item_total_including_tax": "number",
        "table_total_excluding_tax": "number",
        "table_total_tax": "number",
        "table_total_including_tax": "number",
        "company_department_name": "",
        "issued_date": "date",
        "delivery_date": "date",
        "payment_date": "date",
        "invoice_number": "",
        "document_number": "",
        "item_line_number": "number",
    }
}
import sys
sys.path.append(r"D:\Workspace\cinnamon\code\dev\utilities")
from basic_utils import visualize, imread, load_json
import glob
import os
if __name__ == "__main__":
    # parameters
    dataset = "bourbon"
    settype = "train"
    # test on bourbon
    data_path = r"D:\Workspace\cinnamon\data\Bourbon"
    data_list = glob.glob(data_path + "/*")
    for data in data_list:
        # filter for settype
        if settype not in data.lower(): continue

        # fetch label list
        labels_path = os.path.join(data, "labels")
        label_list = glob.glob(labels_path + "/*.json")

        # iteration on label file
        for label_path in label_list:
            # load label
            label = load_json(label_path, only=["key", "value"])

            # get mapping list
            map_dict = mapping_list.get(dataset)

            for item in label:
                print(f"{item.get('type')} ==> {map_dict.get(item.get('type'))}")

            
            break


    

    pass