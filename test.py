import os
import openai
from openai import OpenAI

# api_key_path = os.path.join('..', 'openai_api_key.txt')
# with open(api_key_path, "r") as f:
#     api_key = f.read().strip()
    

# chat_client = OpenAI(api_key=api_key)
# response = chat_client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "This is a test"},
#                     {"role": "user", "content": "Hello, how are you?"},
                    
#                 ]
#                 )
# response = response.choices[0].message.content
# print(response)




def generate_report(config, curr_bus_data, bus_review_summary_dict: dict, other_review_summary_dict: dict, 
                    actions: str) -> None:
    # generate a pdf report with review summaries and actionable insights
    business_id = curr_bus_data.business_id.values[0]
    business_name = curr_bus_data.name.values[0]
    city = curr_bus_data.city.values[0]
    category = curr_bus_data.categories.values[0]
    
    report = f"Business ID: {business_id}\n\
                Business Name: {business_name}\n\
                City: {city}\n\
                Category: {category}\n\
                \n\
                Business' review summaries: \n\
                    \tPositive: {bus_review_summary_dict['pos_summary']};\n\
                    \tNegative: {bus_review_summary_dict['neg_summary']}.\n\
                Other Business' review summaries: \n\
                    \tPositive: {other_review_summary_dict['pos_summary']};\n\
                    \tNegative: {other_review_summary_dict['neg_summary']}.\n\
                \n\
                Actions: {actions}"
    
    # save report to a pdf file
    report_fname = f'{business_name}_review_insights.pdf'
    report_save_path = os.path.join(config['output_dir'], report_fname)
    with open(report_save_path, 'w') as f:
        f.write(report)
    print("Report saved to: ", report_save_path)
    



actions = ''
with open('actions.txt', 'r', encoding='utf-8') as f:
    actions = f.read()
# print(actions)

# generate_report()

business_name = 'test'
report = actions
# save report to a pdf file
report_fname = f'{business_name}_review_insights.pdf'
report_save_path = report_fname
with open(report_save_path, 'w') as f:
        f.write(report)
print("Report saved to: ", report_save_path)
