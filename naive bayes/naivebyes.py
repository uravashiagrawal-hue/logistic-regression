import numpy as np
import pandas as pd
data = pd.read_csv("naive bayes\play_tennis.csv")
print(data.head())
data.drop(columns=['day'], inplace = True)
print(data)

# question => outlook = sunny, temp = hot, humidity = high, wind = weak
# play or not

# solution  =>
# p(yes| sunny, hot,high,weak) = p(sunny|yes) * p(hot|yes)* p(high|yes) * p(weak|yes)
# p(no |sunny,hot,high,weak) = p(sunny|no) * p(hot|no)* p(high|no) * p(weak|no)


# question 2 => outlook = overcast, temp = cold, humidity = low, wind = weak
# play or no play

# solution = p(yes|overcast,cold,low,weak) = p(overcast|yes) * p(cold|yes) * p(low|yes) * p(weak|yes)
# p(no|overcast,cold,low,weak) = p(overcast|no) * p(cold|no) * p(low|no) * p(weak|no)

# process = how a problem is solve by MAP
# training -> lookup table(dictionary) -> testing

print(data['play'].value_counts())
py = 9/14
pn = 5/14

# outlook
op = pd.crosstab(data['outlook'], data['play'])
print(op)
pon = 0
prn = 2/5
psn = 3/5

poy = 4/9
pry = 3/9
psy = 2/9

tp = pd.crosstab(data['temp'], data['play'])
print(tp)
pcooln = 1/5
photn = 2/5
pmildn = 2/5

pcooly = 3/9
photy = 2/9
pmildy = 4/9

# humidity
hp = pd.crosstab(data['humidity'], data['play'])
print(hp)
phighn = 4/5
pnormaln = 1/5

phighy = 3/9
pnormaly = 6/9

# wind
wp = pd.crosstab(data['wind'], data['play'])
print(wp)

pstrongn = 3/5
pweakn = 2/5

pstrongy = 3/9
pweaky = 6/9

# sunny, temp = hot, humidity = high, wind = weak
pyes= py * psy * photy* phighy* pweaky
pno = pn* psn* photn* phighn* pweakn
print(pyes)
print(pno)
