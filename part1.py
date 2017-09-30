import math
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
import scipy.stats as ss
#data set
Type = ["medical","fashion","electrical","government","Property","funiture","educational",
"jewellery","jewellery","medical","fashion","fashion","fashion","fashion","medical",
"fashion","jewellery","government","government","electrical","educational","electrical",
"electrical","fashion","funiture","electrical","Property","electrical","fashion","fashion",
"fashion","electrical","funiture","medical","jewellery","educational","fashion","supermarket",
"jewellery","fashion","educational","fashion","supermarket","fashion","book","medical",
"supermarket","fashion","book","gaming","sports equipment","fashion","gaming","gaming",
"government","funiture","medical","electrical","educational","gaming","Property","book",
"fashion","fashion","jewellery","medical","government","Property","fashion","gaming",
"fashion","educational","Property","fashion","fashion","educational","sports equipment",
"gaming","educational","gaming","sports equipment","electrical","fashion","fashion",
"jewellery","Property","sports equipment","educational","medical","supermarket","book",
"electrical","Property","sports equipment","educational","electrical","fashion","fashion",
"fashion","educational"]

no_of_live_days = [502,189,76,518,423,234,155,543,1009,321,176,298,50,256,33,200,653,86,
180,321,176,298,50,256,33,76,484,111,762,154,600,965,175,96,321,145,189,7,514,654,980,136,
79,865,973,145,335,75,34,28,168,94,54,3,100000,264,433,42,354,87,522,1143,56,438,9,321,176,
298,50,256,33,632,976,58,164,76,54,321,87,609,99,172,743,465,23,2,77,91,76,83,143,532,176,
996,436,78,185,365,909,307]

down_time=[80.3,51.2,27.3,64.3,75.4,5.1,31.4,64.2,101.3,61.3,24.5,43.8,0.5,31.42,1.3,18,126.8,
4.7,13.2,47.1,12.2,41.5,0.3,31.4,1.1,4.1,86.2,3.2,152.9,6.9,114,201.6,12,2.9,47.1,4.8,3.2,2.9,
93.4,126.9,205.2,3.64,5.1,177.6,203.52,4.8,1020,3.2,0,0,10.32,0.6,1,0,56.1,33,73.9,0.8,54.9,
10.7,95.28,244.32,0.3,75.12,9,47,12.24,41.52,8,31.44,0.6,121.7,204.24,5.3,10.8,5.7,3.4,23.1,
2,116.2,5.3,11.3,148.3,81.6,3.3,0,34.1,4.5,31.6,21.6,4.3,91.2,8,212,75,9,14.4,57.6,188.2,43.7]

no_of_hits_missing = [3000,12663,4921,34798,28341,11678,10890,56785,57609,21507,51792,29966,3350,17131,
2221,13454,43789,6000,12067,21507,11792,19966,350,7152,2211,4311,124280,7437,51054,10318,40200,
54444,275,6431,21509,1009,5963,4489,34438,43818,65660,9112,5293,57955,65191,8767,22441,5025,2278,
1876,11256,6298,3618,201,567,17688,29011,2678,23576,5892,34974,345,3752,29346,609,21507,
11792,10099,3350,17152,2211,42344,76543,3886,10966,"NA",3651,21560,5878,40890,6632,11456,49879,
31145,1541,134,5159,6097,5092,5561,9581,356,11798,66732,29212,5226,12000,2456,"NA",3233]

still_alive=["yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes",
"yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes",
"yes","yes","yes","yes","no","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes",
"yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes",
"yes","no","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes",
"yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","yes","no",
"yes","yes","yes","no","yes","yes","yes","yes"]

no_of_sales_missing = [3960,886,1273,4176,3400,1401,762,8919,23043,5376,3625,1397,1340,2056,
555,1608,420,720,4827,2580,2948,1397,42,858,265,1724.4,"NA",892,3573,1238,30150,6534,33,
450,2581,403,1490,538.68,2410,32864,2000,3644,635,6954,7823,6575,1571,603,200,1350,756,6000,
434,175,381,23000,21700,187,321,28929,0,4199,3700,7337,3522,"NA",0,156,402,"NA",884,296,0,272,
1315,23,256,16170,4408,4907,798,1374,3481,3737,20003,0,619,427,1273,667,24,2,0,7689,7303,1306,840,
172,8700,388]

average_sales_value_missing = [15.67,19.99,398.1,1298.76,567.5,676.5,5.5,23.56,578.15,14.8,250.17,40.8,
45.9,30.5,25.7,1056.89,75.5,50.9,77.5,156.9,300.1,98.5,567.14,23.6,256.75,276.75,0,176.5,
23.84,124.5,673,1890,25000,32,75.9,15.99,324.7,45.67,129.99,98.1,150.99,67.5,76.5,43.6,23.56,
78.15,14.8,250.17,30.8,45.9,1300.5,25.7,256.89,25.5,19.99,345,54.1,231,45,76.3,0,45.5,67.1,
198.8,35.6,25.2,0,435,45,"NA",87.34,24,0,67,23.4,550,1345.87,33.3,23.4,92.1,499.99,56.5,67.8,
34.2,234.4,0,155.42,78.9,50.5,141.3,54.67,4230.12,0,287,25,643,25,154.98,543.15,50]

average_user_age=[56,27,45,54,25,46,50,32,46,54,33,23,20,24,37,51,25,48,49,
33,43,47,53,19,31,26,43,23,22,22,38,43,34,19,33,39,41,35,22,24,25,28,56,65,46,
55,20,29,18,23,20,18,37,51,25,43,49,68,23,45,31,52,23,43,53,78,39,51,32,54,
23,38,58,31,21,22,27,43,42,53,23,33,23,20,19,37,51,25,43,49,33,43,42,53,43,56,183,19,31,5]

usability_rating = [1,2,2,4,1,2,3,3,4,4,2,4,3,2,4,2,2,2,3,3,2,1,2,1,2,3,3,2,4,1,2,
3,1,1,2,1,3,1,2,4,1,4,2,3,2,4,1,4,2,4,3,3,2,3,3,4,4,1,1,4,4,2,4,3,
4,3,2,1,4,3,2,1,4,3,1,2,4,4,1,3,3,3,3,3,2,4,1,2,3,3,2,3,1,4,2,2,1,2,3,1]

#functions
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

def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()if count == max_count]

def quantile(x, p):
	p_index = int(p * len(x))
	return sorted(x)[p_index]

def variance(x):
    n= len(x)
    avg = mean(x)
    deviations = sum([(i-avg)*(i-avg) for i in x ])
    return deviations/n
#def standard_deviation(x):
#    return math.sqrt(sum([(i-mean(x))**2 for i in x]) / len(x))

def standard_deviation(x):
	return math.sqrt((variance(x)))

def data_range(x):
	return max(x) - min(x)

def interquartile_range(x):
 	return quantile(x, 0.75) - quantile(x, 0.25)

#identify and remove missing data
no_of_hits = [x for x in no_of_hits_missing if x != "NA" ]
no_of_sales = [x for x in no_of_sales_missing if x != "NA"]
average_sales_value = [x for x in average_sales_value_missing if x != "NA"]

#generating descriptive statistics and identify outliers with plots
print("Discriptive statistics - type_of_website: ")
his_type = Counter(Type)
print(his_type)
labels = list(his_type.keys())
sizes = list(his_type.values())
colors=['green','red','yellow','blue','purple','lightskyblue','lightcoral','orchid','grey','orange','pink']
patches, texts = plt.pie(sizes, colors=colors,shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()

#plt.bar([x*2 for x in range(len(his_type.keys()))],his_type.values(),align='center')
#plt.xticks([x*2 for x in range(len(his_type))], list(his_type.keys()))
#plt.yticks([x*2 for x in range(16)])
#plt.xlabel("Type of Website")
#plt.ylabel("No. of website")
#plt.title("Histogram of Type of Website")
#plt.show()

print("Discriptive statistics - no_of_live_days: ")
no_of_live_days_mean = mean(no_of_live_days)
no_of_live_days_median = median(no_of_live_days)
no_of_live_days_mode = mode(no_of_live_days)
no_of_live_days_range = data_range(no_of_live_days)
no_of_live_days_IQR = interquartile_range(no_of_live_days)
no_of_live_days_variance = variance(no_of_live_days)
no_of_live_days_SD = standard_deviation(no_of_live_days)
print("Mean = ",no_of_live_days_mean)
print("Median = ",no_of_live_days_median)
print("Mode = ",no_of_live_days_mode)
print("Range = ",no_of_live_days_range)
print("IQR = ",no_of_live_days_IQR)
print("Variance = ",no_of_live_days_variance)
print("Standard Deviation = ",no_of_live_days_SD)
print("By calculating mild fence (",round(quantile(no_of_live_days,0.25)-1.5*interquartile_range(no_of_live_days),4)," , ",round(quantile(no_of_live_days,0.75)+1.5*interquartile_range(no_of_live_days),4),") using IQR, any value that falls out this fence is considered as an outlier.")
no_of_live_days_outlier = [x for x in no_of_live_days if x>quantile(no_of_live_days,0.75)+1.5*interquartile_range(no_of_live_days) or x<quantile(no_of_live_days,0.25)-1.5*interquartile_range(no_of_live_days)]
print("no_of_live_days_outlier: ",no_of_live_days_outlier)
plt.hist(no_of_live_days)
plt.title("Histogram of No. of live days")
plt.xlabel("No. of live days (unit: day)")
plt.ylabel("# of websites")
plt.show()


print("\nDiscriptive statistics - down_time: ")
down_time_mean = mean(down_time)
down_time_median = median(down_time)
down_time_mode = mode(down_time)
down_time_range = data_range(down_time)
down_time_IQR = interquartile_range(down_time)
down_time_variance = variance(down_time)
down_time_SD = standard_deviation(down_time)
print("Mean = ",down_time_mean)
print("Median = ",down_time_median)
print("Mode = ",down_time_mode)
print("Range = ",down_time_range)
print("IQR = ",down_time_IQR)
print("Variance = ",down_time_variance)
print("Standard Deviation = ",down_time_SD)
print("By calculating mild fence (",round(quantile(down_time,0.25)-1.5*interquartile_range(down_time),4)," , ",round(quantile(down_time,0.75)+1.5*interquartile_range(down_time),4),") using IQR, any value that falls out this fence is considered as an outlier.")
down_time_outlier = [x for x in down_time if x>quantile(down_time,0.75)+1.5*interquartile_range(down_time) or x<quantile(down_time,0.25)-1.5*interquartile_range(down_time)]
print("down_time_outlier: ",down_time_outlier)
down_time1 = [ x/10 for x in down_time]
plt.hist(down_time1)
plt.xticks([x*10 for x in range(0,max(down_time)//100+1) ])
plt.yticks([x*5 for x in range(0,21)])
plt.title("Histogram of Down time")
plt.xlabel("Down time (unit: 10 hrs)")
plt.ylabel("# of websites")
plt.show()


print("\nDiscriptive statistics - no_of_hits: ")
no_of_hits_mean = mean(no_of_hits)
no_of_hits_median = median(no_of_hits)
no_of_hits_mode = mode(no_of_hits)
no_of_hits_range = data_range(no_of_hits)
no_of_hits_IQR = interquartile_range(no_of_hits)
no_of_hits_variance = variance(no_of_hits)
no_of_hits_SD = standard_deviation(no_of_hits)
print("Mean = ",no_of_hits_mean)
print("Median = ",no_of_hits_median)
print("Mode = ",no_of_hits_mode)
print("Range = ",no_of_hits_range)
print("IQR = ",no_of_hits_IQR)
print("Variance = ",no_of_hits_variance)
print("Standard Deviation = ",no_of_hits_SD)
print("By calculating mild fence (",round(quantile(no_of_hits,0.25)-1.5*interquartile_range(no_of_hits),4)," , ",round(quantile(no_of_hits,0.75)+1.5*interquartile_range(no_of_hits),4),") using IQR, any value that falls out this fence is considered as an outlier.")
no_of_hits_outlier = [x for x in no_of_hits if x>quantile(no_of_hits,0.75)+1.5*interquartile_range(no_of_hits) or x<quantile(no_of_hits,0.25)-1.5*interquartile_range(no_of_hits)]
print("no_of_hits_outlier: ",no_of_hits_outlier)
plt.hist(no_of_hits)
plt.title("Histogram of No. of hits")
plt.xlabel("No. of hits")
plt.ylabel("# of websites")
plt.show()


print("Discriptive statistics - websit still alive: ")
his_live = Counter(still_alive)
print(his_live)
plt.bar(range(len(his_live.keys())),his_live.values(),align='center')
plt.xticks(range(len(his_live)), list(his_live.keys()))
plt.yticks([x*5 for x in range(21)])
plt.xlabel("Still alive")
plt.ylabel("No. of website")
plt.title("Histogram of Website still alive")
plt.show()

print("\nDiscriptive statistics - no_of_sales: ")
no_of_sales_mean = mean(no_of_sales)
no_of_sales_median = median(no_of_sales)
no_of_sales_mode = mode(no_of_sales)
no_of_sales_range = data_range(no_of_sales)
no_of_sales_IQR = interquartile_range(no_of_sales)
no_of_sales_variance = variance(no_of_sales)
no_of_sales_SD = standard_deviation(no_of_sales)
print("Mean = ",no_of_sales_mean)
print("Median = ",no_of_sales_median)
print("Mode = ",no_of_sales_mode)
print("Range = ",no_of_sales_range)
print("IQR = ",no_of_sales_IQR)
print("Variance = ",no_of_sales_variance)
print("Standard Deviation = ",no_of_sales_SD)
print("By calculating mild fence (",round(quantile(no_of_sales,0.25)-1.5*interquartile_range(no_of_sales),4)," , ",round(quantile(no_of_sales,0.75)+1.5*interquartile_range(no_of_sales),4),") using IQR, any value that falls out this fence is considered as an outlier.")
no_of_sales_outlier = [x for x in no_of_sales if x>quantile(no_of_sales,0.75)+1.5*interquartile_range(no_of_sales) or x<quantile(no_of_sales,0.25)-1.5*interquartile_range(no_of_sales)]
print("no_of_sales_outlier: ",no_of_sales_outlier)
decile1 = lambda x: x//1000*1000
his_sales = Counter([decile1(x) for x in no_of_sales])
print(his_sales)
x_sales = list(his_sales.keys())
y_sales = list(his_sales.values())
plt.bar([x+0.4 for x in range(len(y_sales))],y_sales,)
#plt.bar([x+4 for x in range(len(his_sales.keys()))],his_sales.values())
#plt.xticks([x*1500 for x in range(21)])
#plt.xlabel("Type of Website")
#plt.ylabel("No. of website")
#plt.title("Histogram of Type of Website")
#plt.show()



#plt.hist(no_of_sales)
plt.title("Histogram of No. of sales")
plt.xlabel("No. of sales")
plt.ylabel("# of websites")
plt.show()


print("\nDiscriptive statistics - average_sales_value: ")
average_sales_value_mean = mean(average_sales_value)
average_sales_value_median = median(average_sales_value)
average_sales_value_mode = mode(average_sales_value)
average_sales_value_range = data_range(average_sales_value)
average_sales_value_IQR = interquartile_range(average_sales_value)
average_sales_value_variance = variance(average_sales_value)
average_sales_value_SD = standard_deviation(average_sales_value)
print("Mean = ",average_sales_value_mean)
print("Median = ",average_sales_value_median)
print("Mode = ",average_sales_value_mode)
print("Range = ",average_sales_value_range)
print("IQR = ",average_sales_value_IQR)
print("Variance = ",average_sales_value_variance)
print("Standard Deviation = ",average_sales_value_SD)
print("By calculating mild fence (",round(quantile(average_sales_value,0.25)-1.5*interquartile_range(average_sales_value),4)," , ",round(quantile(average_sales_value,0.75)+1.5*interquartile_range(average_sales_value),4),") using IQR, any value that falls out this fence is considered as an outlier.")
average_sales_value_outliers = [x for x in average_sales_value if x>quantile(average_sales_value,0.75)+1.5*interquartile_range(average_sales_value) or x<quantile(average_sales_value,0.25)-1.5*interquartile_range(average_sales_value)]
print("average_sales_value_outliers: ",average_sales_value_outliers)
plt.hist(average_sales_value)
plt.title("Histogram of average_sales_value")
plt.xlabel("average_sales_value (Unit: $)")
plt.ylabel("# of websites")
plt.show()


print("\nDiscriptive statistics - average_user_age: ")
average_user_age_mean = mean(average_user_age)
average_user_age_median = median(average_user_age)
average_user_age_mode = mode(average_user_age)
average_user_age_range = data_range(average_user_age)
average_user_age_IQR = interquartile_range(average_user_age)
average_user_age_variance = variance(average_user_age)
average_user_age_SD = standard_deviation(average_user_age)
print("Mean = ",average_user_age_mean)
print("Median = ",average_user_age_median)
print("Mode = ",average_user_age_mode)
print("Range = ",average_user_age_range)
print("IQR = ",average_user_age_IQR)
print("Variance = ",average_user_age_variance)
print("Standard Deviation = ",average_user_age_SD)
print("By calculating mild fence (",round(quantile(average_user_age,0.25)-1.5*interquartile_range(average_user_age),4)," , ",round(quantile(average_user_age,0.75)+1.5*interquartile_range(average_user_age),4),") using IQR, any value that falls out this fence is considered as an outlier.")
average_user_age_outliers = [x for x in average_user_age if x>quantile(average_user_age,0.75)+1.5*interquartile_range(average_user_age) or x<quantile(average_user_age,0.25)-1.5*interquartile_range(average_user_age)]
print("average_user_age_outliers: ",average_user_age_outliers)
plt.hist(average_user_age)
plt.title("Histogram of average_user_age")
plt.xlabel("average_user_age")
plt.ylabel("# of websites")
plt.show()

print("\nDiscriptive statistics - usability_rating: ")
usability_rating_mean = mean(usability_rating)
usability_rating_median = median(usability_rating)
usability_rating_mode = mode(usability_rating)
usability_rating_range = data_range(usability_rating)
usability_rating_IQR = interquartile_range(usability_rating)
usability_rating_variance = variance(usability_rating)
usability_rating_SD = standard_deviation(usability_rating)
print("Mean = ",usability_rating_mean)
print("Median = ",usability_rating_median)
print("Mode = ",usability_rating_mode)
print("Range = ",usability_rating_range)
print("IQR = ",usability_rating_IQR)
print("Variance = ",usability_rating_variance)
print("Standard Deviation = ",usability_rating_SD)
print("By calculating mild fence (",round(quantile(usability_rating,0.25)-1.5*interquartile_range(usability_rating),4)," , ",round(quantile(usability_rating,0.75)+1.5*interquartile_range(usability_rating),4),") using IQR, any value that falls out this fence is considered as an outlier.")
usability_rating_outlier = [x for x in usability_rating if x>quantile(usability_rating,0.75)+1.5*interquartile_range(usability_rating) or x<quantile(usability_rating,0.25)-1.5*interquartile_range(usability_rating)]
print("usability_rating_outlier: ",usability_rating_outlier)
plt.hist(usability_rating,align='mid')
plt.xticks([x for x in range(1,5)])
plt.title("Histogram of usability_rating")
plt.xlabel("usability_rating")
plt.ylabel("# of websites")
plt.show()


#correlation & linear regression
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

def b1(x,y):
    x_mean = mean(x)
    y_mean = mean(y)
    n =len(x)
    num = sum([j*k for j,k in zip(x,y)])-n*x_mean*y_mean
    den = sum([i*i for i in x])-n*x_mean*x_mean

    #return (standard_deviation(y)/standard_deviation(x))*pearson_correlation(x,y)
    return num/den

def b0(x,y):
	return mean(y) - b1(x,y)*mean(x)

def linearRegression(m,c,x):
    y=m*x+c
    return y

def get_x(m,c,y):
	x = (y-c)/m
	return x


print("Correlation & Regression")
#no_of_live_days & no_of_hits
print("#no_of_live_days & no_of_hits")

def removeMissing(x,y):
    list=[]
    for i in range(len(y)):
        if y[i] != "NA" :
            list.append(x[i])
    return list

no_of_live_days_h = removeMissing(no_of_live_days,no_of_hits_missing)

m = b1(no_of_live_days_h,no_of_hits)
c = b0(no_of_live_days_h,no_of_hits)

predictedNo_of_hits_m = [linearRegression(m,c,x) for x in no_of_hits]
residuals = [x-y for x,y in zip(no_of_hits,predictedNo_of_hits_m)]

#[x for no_of_live_days[i] for i in enumerate(no_of_hits_missing) if no_of_hits_missing[i] != "NA"]
plt.scatter(no_of_live_days_h,no_of_hits)

plt.xlabel("No. of live days")
plt.ylabel("No. of hits")
plt.title("No. of live days VS. No. of hits")
plt.xlim(0,1000)
plt.show()

r1 = pearson_correlation(no_of_live_days_h,no_of_hits)
print("r^2 = ",round(r1**2,4))

print("Correlation r = ",round(r1,4))
print("The least linear regression line of these two variables is y = ",round(m,4),"x + ",round(c,4))
plt.scatter(no_of_live_days_h,no_of_hits)
axes = plt.gca()
m, b = np.polyfit(no_of_live_days_h,predictedNo_of_hits_m, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b,'g-')
plt.xlabel("No. of live days")
plt.ylabel("No. of hits")
plt.title("No. of live days VS. No. of hits")
plt.xlim(0,1000)
plt.show()


#no_of_live_days & no_of_salse
print("#no_of_live_days & no_of_salse")

no_of_live_days_s= removeMissing(no_of_live_days,no_of_sales_missing)

m1 = b1(no_of_live_days_s,no_of_sales)
c1 = b0(no_of_live_days_s,no_of_sales)
predictedNo_of_sales_m = [linearRegression(m1,c1,x) for x in no_of_hits]
residuals = [x-y for x,y in zip(no_of_sales,predictedNo_of_sales_m)]

plt.scatter(no_of_live_days_s,no_of_sales)

plt.xlabel("No. of live days")
plt.ylabel("No. of hits")
plt.title("No. of live days VS. No. of hits")
plt.xlim(0,1000)
plt.show()

r2 = pearson_correlation(no_of_live_days_s,no_of_sales)

print("Correlation r = ",round(r2,4))
print("r^2 = ",round(r2**2,4))
print("The least linear regression line of these two variables is y = ",round(m1,4),"x + ",round(c1,4))

plt.scatter(no_of_live_days_s,no_of_sales)
axes = plt.gca()
m, b = np.polyfit(no_of_live_days_s,no_of_sales, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b,'g-')
plt.xlabel("No. of live days")
plt.ylabel("No. of salse")
plt.title("No. of live days VS. No. of sales")
plt.xlim(0,1000)
plt.show()


#no_of_salse & average_sales_value
print("#no_of_salse & average_sales_value")
average_sales_value_s= removeMissing(average_sales_value_missing,no_of_sales_missing)

m2 = b1(no_of_sales,average_sales_value_s)
c2 = b0(no_of_sales,average_sales_value_s)
predictedAverage_sales_value = [linearRegression(m2,c2,x) for x in average_sales_value_s]
residuals = [x-y for x,y in zip(average_sales_value_s,predictedAverage_sales_value)]

plt.scatter(no_of_sales,average_sales_value_s)

plt.xlabel("No. of sales")
plt.ylabel("Average sales value")
plt.title("No. of sales VS. Average sales value")
plt.ylim(0,5000)
plt.show()

r3 = pearson_correlation(no_of_sales,average_sales_value_s)
print("Correlation r = ",round(r3,4))
print("r^2 = ",round(r3**2,4))

print("The least linear regression line of these two variables is y = ",round(m2,4),"x + ",round(c2,4))

plt.scatter(no_of_sales,average_sales_value_s)
axes = plt.gca()
m, b = np.polyfit(no_of_sales,average_sales_value_s, 1)
X_plot = np.linspace(axes.get_xlim()[0],axes.get_xlim()[1],100)
plt.plot(X_plot, m*X_plot + b,'g-')
plt.xlabel("No. of sales")
plt.ylabel("Average sales value")
plt.ylim(0,5000)

plt.title("No. of sales VS. Average sales value")
plt.show()

def confidence_interval(mean,stdev,n,a,s):
    z = abs(round(ss.norm.ppf(q=(1-a)/2),4))
    s_e = round(stdev/math.sqrt(n),4)
    print(round(a*100,0),"% -> z = ",z)
    print("Standard Error = stdev / √n = ",s_e)
    x_upper = mean + z*s_e
    x_lower = mean - z*s_e
    print("Point Estimates  = mean +/- z * Standard error -> (",round(x_lower,4)," , ",round(x_upper,4)," )" )
    print("So, We're ",round(a*100,0),"% sure that the population mean of ",s," are between ",round(x_lower,4)," and ",round(x_upper,4),".")


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

#hypothesis test 1
print("#confidence interval1")
confidence_interval(usability_rating_mean,usability_rating_SD,len(usability_rating),0.95,"Usability Rating")
print("#hypothesis test 1")
h0_1 = "The normal mean of usability rating of a website is 3 in 2016."
ha_1 = "The normal mean of usability rating of a website is not 3 in 2016."
Hypothesis_testing_two_tail(3,usability_rating_mean,usability_rating_SD,len(usability_rating),0.95,h0_1)


#hypothesis test 2
print("#confidence interval2")
confidence_interval(down_time_mean,down_time_SD,len(down_time),0.99,"Down Time")
print("#hypothesis test 2")
h0_2 = "The average down time of a website in 2016 is less than or equal to 60 hours."
ha_2 = "The average down time of a website in 2016 is greater than 60 hours."
Hypothesis_testing_right_tail(60,down_time_mean,down_time_SD,len(down_time),0.99,h0_2)

#hypothesis test 3
print("#confidence interval3")
confidence_interval(average_sales_value_mean,average_sales_value_SD,len(average_sales_value),0.98,"Average sales value")
print("#hypothesis test 3")
h0_3 = "The average sales value of a website in 2016 is greater than or equal to $500."
ha_3 = "The average sales value of a website in 2016 is less than $500."
Hypothesis_testing_left_tail(500,average_sales_value_mean,average_sales_value_SD,len(average_sales_value),0.98,h0_3)
