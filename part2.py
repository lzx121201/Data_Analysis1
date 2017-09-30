from __future__ import division
from collections import Counter
import math, random, csv, json
import os
import ast
from bs4 import BeautifulSoup
import requests
import datetime
import math
from matplotlib import pyplot as plt
from collections import Counter
import scipy.stats as ss

ftweets = []
tweets_file = []

def read_ftweets():
	with open(os.path.expanduser("~/Desktop/DataA/DA_final_project/tweets1.txt"), 'r' , encoding= 'ISO-8859-1') as f:
		for line in f:
			ftweets.append(line)

def convert_ftweets_to_dict():
	for i in range(0, len(ftweets)):
		tweets_file.append(ast.literal_eval(ftweets[i]))

def extrac_num_of_tweets(x):
	return [x[key]['user']['statuses_count'] for key in range(0,len(x))]

def extrac_date_of_created(x):
	return [x[key]['user']['created_at'] for key in range(0,len(x))]

def day_difference(date):
	current=datetime.datetime.today()
	aim=datetime.datetime(datetime.datetime.strptime(date,"%a %b %d %H:%M:%S %z %Y").year,datetime.datetime.strptime(date,"%a %b %d %H:%M:%S %z %Y").month,datetime.datetime.strptime(date,"%a %b %d %H:%M:%S %z %Y").day,0,0)
	return (current-aim).days

def date_difference_day(x):
	return [day_difference(x[key]['user']['created_at']) for key in range(0,len(x))]

def calculate_avg(x,y):
	if x != 0 and y != 0:
		return x/y
	else:
		return 0

def avg_post_per_day(x):
	return [ calculate_avg(x[0][i],x[1][i]) for i in range(0,len(x[1]))]

def setup_matrix(m1,m2):
	return [[m1[i] for i in range(len(m1))],[m2[h] for h in range(len(m2))]]

read_ftweets()
convert_ftweets_to_dict()

no_of_tweets=extrac_num_of_tweets(tweets_file)

#date=date_of_created[0][1]
#print(date)
#current=datetime.datetime.today()
#print(current)
#aim=datetime.datetime(datetime.datetime.strptime(date,"%a %b %d %H:%M:%S %z %Y").year,datetime.datetime.strptime(date,"%a %b %d %H:%M:%S %z %Y").month,datetime.datetime.strptime(date,"%a %b %d %H:%M:%S %z %Y").day,0,0)
#print((current-aim).days)
day_differences = date_difference_day(tweets_file)

data_set = setup_matrix(no_of_tweets,day_differences)

data = avg_post_per_day(data_set)

plt.hist(data)
plt.show()
def dot(x, y):
	return sum(x_i * y_i for x_i, y_i in zip(x, y))

def mean(x):
    return sum(x)/len(x)

def median(x):
    n = len(x)
    sorted_x = sorted(x)
    midpoint = n //2
    if n % 2 == 1:
        return sorted_x[midpoint]
    else:
        return (sorted_x[midpoint-1]+sorted_x[midpoint]) /2

def quantile(x, p):
	p_index = int(p * len(x))
	return sorted(x)[p_index]

def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()if count == max_count]

def x_minus_mean(x):
 	x_bar = mean(x)
 	return [x_i - x_bar for x_i in x]

from statistics import stdev


def variance(x):
    n= len(x)
    avg = mean(x)
    deviations = sum([(i-avg)*(i-avg) for i in x ])
    return deviations/n

def standard_deviation(x):
	return math.sqrt((variance(x)))

def data_range(x):
	return max(x) - min(x)

def interquartile_range(x):
 	return quantile(x, 0.75) - quantile(x, 0.25)


print("Mild fence: (",quantile(data,0.25)-1.5*interquartile_range(data),",",quantile(data,0.75)+1.5*interquartile_range(data),")")

print("Before removing outlier:")
print("Mode:",mode(data))
print("Median:",median(data))
print("Mean:",mean(data))
print("Range:",data_range(data))
print("Interquartile Range:",interquartile_range(data))
print("Standard Deviation:",standard_deviation(data))

no_outliers = [x for x in data if x<quantile(data,0.75)+1.5*interquartile_range(data) and x>quantile(data,0.25)-1.5*interquartile_range(data)]
print(max(no_outliers),min(no_outliers))
print(len(no_outliers))

print("After removing outlier:")
print("Mode:",mode(no_outliers))
print("Median:",median(no_outliers))
print("Mean:",mean(no_outliers))
print("Range:",data_range(no_outliers))
print("Interquartile Range:",interquartile_range(no_outliers))
print("Standard Deviation:",standard_deviation(no_outliers))

decile = lambda x: x //10*10
histogram = Counter( decile (x) for x in no_outliers)
print(histogram)
plt.bar([x+5 for x in histogram.keys()], histogram.values(),10)
plt.xticks([x*10 for x in range(11) ])
plt.yticks([x*50 for x in range(21) ])
plt.xlabel("# Tweets/Day")
plt.ylabel("# of users")
plt.title("The frequency of Number of Tweets Per Day")
plt.ylim(0,1000)
plt.xlim(0,100)
plt.show()

""" Hypothesis Testing """
def confidence_interval(mean,stdev,n,a,s):
    z = abs(round(ss.norm.ppf(q=(1-a)/2),4))
    s_e = round(stdev/math.sqrt(n),4)
    print(round(a*100,0),"% -> z = ",z)
    print("Standard Error = stdev / √n = ",s_e)
    x_upper = mean + z*s_e
    x_lower = mean - z*s_e
    print("Point Estimates  = mean +/- z * Standard error -> (",round(x_lower,4)," , ",round(x_upper,4)," )" )
    print("So, We're ",round(a*100,0),"% sure that the population mean of ",s," are between ",round(x_lower,4)," and ",round(x_upper,4),".")


def pearson_correlation(x,y):
    x_mean = mean(x)
    y_mean = mean(y)
    n =len(x)
    num = sum([j*k for j,k in zip(x,y)])-n*x_mean*y_mean
    den = math.sqrt((sum([i**2 for i in x])-n*x_mean*x_mean)*(sum([z**2 for z in y])-n*y_mean*y_mean))
#    s_x = standard_deviation(x)
#    s_y = standard_deviation(y)
#    x_mean = mean(x)
#    y_mean = mean(y)
#    n = len(x)
#    return (sum_xy-n*x_mean*y_mean)/(s_x*s_y)
    return num/den

def Hypothesis_testing_two_tail(claim,mean,stdev,n,a,h0):
    z = abs(round(ss.norm.ppf(q=(1-a)/2),4))
    test_stats = (mean - claim) / (stdev/math.sqrt(n))
    print("test statistics = ( Observed Value - Hypothesised Vlaue ) / Standard Error")
    print("\t = ( ",round(mean,4)," - ",claim," ) / ( ",round(stdev,4)," / √ ",n," ) ")
    print("\t =",round(test_stats,4))
    result = -1*z < test_stats and test_stats < z
    print(round((1-a)*100,0),"% Significance Level ( " ,-1*z," , ",z," )")
    print(-1*z," < ",round(test_stats,4),"(test statistic) < ",z," (",result,") ")
    if result == True :
        print( "We fail to reject the null hypothesis and conclude that there is insufficient evidence to reject the claim that " + h0)
    else:
        print("The null hypothesis is rejected and there is sufficient evidence to warrant reject of the claim that "+h0)

def Hypothesis_testing_right_tail(claim,mean,stdev,n,a,h0):
    z = abs(round(ss.norm.ppf(q=(1-a)),4))
    test_stats = (mean - claim) / (stdev/math.sqrt(n))
    result = test_stats < z
    print("test statistics = ( Observed Value - Hypothesised Vlaue ) / Standard Error")
    print("\t = ( ",round(mean,4)," - ",claim," ) / ( ",round(stdev,4)," / √ ",n," ) ")
    print("\t =",round(test_stats,4))
    print(round((1-a)*100,0),"% Significance Level")
    print(round(test_stats,4),"(test statistic) < ",z," (",result,") ")
    if result == True :
        print( "We fail to reject the null hypothesis and conclude that there is insufficient evidence to reject the claim that " + h0)
    else:
        print("The null hypothesis is rejected and there is sufficient evidence to warrant reject of the claim that "+h0)

def Hypothesis_testing_left_tail(claim,mean,stdev,n,a,h0):
    z = abs(round(ss.norm.ppf(q=(1-a)),4))
    test_stats = (mean - claim) / (stdev/math.sqrt(n))
    result = test_stats > -1*z
    print("test statistics = ( Observed Value - Hypothesised Vlaue ) / Standard Error")
    print("\t = ( ",round(mean,4)," - ",claim," ) / ( ",round(stdev,4)," / √ ",n," ) ")
    print("\t =",round(test_stats,4))
    print(round((1-a)*100,0),"% Significance Level")
    print(round(test_stats,4),"(test statistic) > ",-1*z," (",result,") ")
    if result == True :
        print( "We fail to reject the null hypothesis and conclude that there is insufficient evidence to reject the claim that " + h0)
    else:
        print("The null hypothesis is rejected and there is sufficient evidence to warrant reject of the claim that "+h0)

print("hypothesis test1")
confidence_interval(mean(no_outliers),standard_deviation(no_outliers),len(no_outliers),0.99,"The average number of tweets per day of a Twitter user who likes Coldplay since the account being created")
h0= "The average number of tweets per day of a Twitter user who likes Coldplay since the account being created is 18."
ha = "The average number of tweets per day of a Twitter user who likes Coldplay since the account being created is not 18."
Hypothesis_testing_two_tail(18,mean(no_outliers),standard_deviation(no_outliers),len(no_outliers),0.99,h0)
