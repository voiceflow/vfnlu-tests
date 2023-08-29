# Running a regression test
To run a regession test pass in the project id, and two version ids along with your api key.
```python
run_regression_test(
    project_id="64cec353cc248300068a77a5",
    version_id1="64ed260c9c1d720007cbff95", 
    version_id2="64cec353cc248300068a77a6", 
    api_key="VF.DM...."
    )
```
You will have the results returned in a dictionary that will have the accuracy, f1 scores, improvements and regressions.

```python
{
    '64ed260c9c1d720007cbff95': 
     {'accuracy': 0.8838289962825279, 'f1': 0.8996408552869768}, 
    '64cec353cc248300068a77a6': 
     {'accuracy': 0.8912639405204461, 'f1': 0.9010508211045153}, 
    'regressions': {'count': 19, 'values': [{'intent': 'alarm_query', 'utterance': 'is there an alarm at four am'}, {'intent': 'calendar_query', 'utterance': 'what is my schedule for tomorrow'}, {'intent': 'calendar_query', 'utterance': 'what do i have scheduled for next week'}, {'intent': 'email_querycontact', 'utterance': 'have you responded my phone calls'}, {'intent': 'general_commandstop', 'utterance': 'olly can you stop it.'}, {'intent': 'general_confirm', 'utterance': 'could you check my last question.'}, {'intent': 'general_dontcare', 'utterance': "wouldn't really know, olly."}, {'intent': 'general_dontcare', 'utterance': 'have no idea regarding to what to select, olly.'}, {'intent': 'general_explain', 'utterance': 'please will you clarify me more on that.'}, {'intent': 'general_negate', 'utterance': "i didn't say it."}, {'intent': 'general_quirky', 'utterance': 'tell me what happens when we die'}, {'intent': 'iot_hue_lightchange', 'utterance': 'can you change the light colors in the house'}, {'intent': 'iot_hue_lightup', 'utterance': 'i need the lights in here to be turned up to seven'}, {'intent': 'lists_remove', 'utterance': 'change that off the list'}, {'intent': 'lists_remove', 'utterance': 'get rid of peas'}, {'intent': 'play_game', 'utterance': 'give me a company for playing football'}, {'intent': 'play_game', 'utterance': 'paper scissors or stone'}, {'intent': 'qa_definition', 'utterance': 'define rumplestiltskin'}, {'intent': 'weather_query', 'utterance': 'are there any tornado warnings today'}]}, 
    'improvements': {'count': 27, 'values': [{'intent': 'calendar_remove', 'utterance': 'remove upcoming task'}, {'intent': 'calendar_remove', 'utterance': "remove emma's birthday from events"}, {'intent': 'cooking_recipe', 'utterance': 'why do people use avocado seeds'}, {'intent': 'datetime_query', 'utterance': 'what is today'}, {'intent': 'email_querycontact', 'utterance': 'does contact mona has an email as well'}, {'intent': 'general_confirm', 'utterance': 'would you please confirm the instruction.'}, {'intent': 'general_explain', 'utterance': 'will you rephrase me about your response once again please.'}, {'intent': 'general_explain', 'utterance': "couldn't understand what you just said now."}, {'intent': 'general_explain', 'utterance': "didn't understand what you said now."}, {'intent': 'general_quirky', 'utterance': 'anything i need to know'}, {'intent': 'general_quirky', 'utterance': 'where did he was yesterday'}, {'intent': 'iot_hue_lightoff', 'utterance': 'switch off main light'}, {'intent': 'iot_wemo_on', 'utterance': 'start new smart socket'}, {'intent': 'music_query', 'utterance': 'who is this'}, {'intent': 'music_settings', 'utterance': 'please repeat that music again of akon'}, {'intent': 'news_query', 'utterance': 'add newscast time to daily schedule'}, {'intent': 'qa_factoid', 'utterance': 'info on lisa ann please'}, {'intent': 'qa_maths', 'utterance': 'fifteen plus twenty'}, {'intent': 'qa_stock', 'utterance': 'fill me in on stock symbol'}, {'intent': 'qa_stock', 'utterance': 'for how much is hp selling for'}, {'intent': 'recommendation_events', 'utterance': 'what event do you suggest for me tonight'}, {'intent': 'recommendation_locations', 'utterance': 'is there a sports bar near kansas city plaza area'}, {'intent': 'recommendation_locations', 'utterance': 'give me some cafes near downtown fort lauderdale'}, {'intent': 'social_post', 'utterance': 'post new status'}, {'intent': 'takeaway_order', 'utterance': 'i want to order food'}, {'intent': 'takeaway_query', 'utterance': 'where are my takeaway order'}, {'intent': 'transport_query', 'utterance': 'when i should leave to office'}]}
}

```