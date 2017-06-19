# speech event identifier
# - 1: speech started
# - 2: speech ended
# - 3: speech recognized
# - 4: silence detected
int8 activity
# the phrase that's just recognized
string phrase
float32 confidence
bool silence