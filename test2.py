import re

with open('actions.txt', 'r', encoding='utf-8') as f:
    actions = f.read()
    
def parse_text(text):
    points = re.split(r'\d+\.\s', text)
    points = [point.strip() for point in points if point.strip()]
    return points

print(actions)
parsed_actions = parse_text(actions)
for action in parsed_actions:
    title, description = action.split(':')
    title = title.replace('**', '')
    description = description.replace('-', '')
    print("Title: ", title)
    print("Description: ", description)
    print("-"*30)