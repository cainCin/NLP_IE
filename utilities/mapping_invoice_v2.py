

mapping_list = {
    "bourbon": {
        "shipping_date": "date",
        "item_number": "number",
        "number_of_cases": "number",
        "number_of_balls_in_case": "number",
        "number_of_pieces": "number",
        "quantity": "quantity",
        "unit": "unit",	
        "order_number": "index",	
        '\t\nitem_number\n': "number",
        '\t\norder_number\n': "index",
        'number_of_cases\n': "number",
        '\norder_number\n': "index",
    },
    "invoice":{
        "company_name":	"",
        "branch_name": "",	
        "company_tel": "",
        "company_fax": "",
        "company_address": "",
        "company_zipcode": "",
        "branch_address": "",
        "branch_zipcode": "",
        "branch_tel": "",
        "branch_fax": "",
        "amount_excluding_tax": "amount_excluding_tax",
        "amount_tax": "amount_tax",
        "tax": "tax",
        "amount_including_tax": "amount_including_tax",
        "bank_name": "",
        "bank_branch_name": "",	
        "account_type": "",
        "account_number": "",
        "account_name": "",
        "item_name": "item_name",
        "item_quantity": "item_quantity",
        "item_quantity_item_unit": "item_quantity",
        "item_unit" : "item_unit",
        "item_unit_amount": "item_unit_amount",
        "item_total_excluding_tax": "item_total_excluding_tax",
        "item_total_tax": "item_total_tax",
        "item_total_including_tax": "item_total_including_tax",
        "table_total_excluding_tax": "amount_excluding_tax",
        "table_total_tax": "amount_tax",
        "table_total_including_tax": "amount_including_tax",
        "company_department_name": "",
        "issued_date": "issued_date",
        "delivery_date": "delivery_date",
        "payment_date": "payment_date",
        "invoice_number": "invoice_number",
        "document_number": "document_number",
        "item_line_number": "item_line_number",
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