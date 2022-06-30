'''
Importing libraries
'''
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import numpy as np
import sys
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import os
from pathlib import Path
import datetime
import calendar
from scipy.stats import linregress

def findDay(test):
    born = datetime.datetime.strptime(test, '%Y-%m-%d').weekday()
    return (calendar.day_name[born])


# Date time conversion registration
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
pd.set_option('mode.chained_assignment', None)

if(os.path.isdir("./barGraphs") == False):os.mkdir("./barGraphs")
if(os.path.isdir("./lineGraphs") == False):os.mkdir("./lineGraphs")
if(os.path.isdir("./scatterGraphs") == False):os.mkdir("./scatterGraphs")

print ("*************************************************************")
print ("Mascarenhas Alexandre")
print ("Experimental Design in Computer Science 2022/1\nUniversity of Tsukuba")
print ("Report 1\n")
print ("Relation between Daily Number of Cyclists and Daily Average")
print ("temperature on the Faria Limas's cycle path in the city of\nSao Paulo, Brazil")
print ("*************************************************************\n")

'''
Plota linha com selecao de dias da semana

'''

plt.rcParams['figure.figsize'] = (12, 6)

'''
importing datasets
'''


naruto = pd.read_csv("faria-lima.csv", delimiter = ',', usecols=['Date', 'Pinheiros']) #import dataset with number of cyclists
bikers = naruto[:-834] #01 jan 2020
#print (bikers)

luffy = pd.read_csv("temperature-sp.csv", delimiter = '\t', usecols=['Data Medicao','TEMPERATURA MEDIA COMPENSADA, DIARIA(°C)'], parse_dates=['Data Medicao']) #import dataset with temperature
luffy.set_index('Data Medicao', inplace = True)
luffy = luffy.rename(columns={"TEMPERATURA MEDIA COMPENSADA, DIARIA(°C)": "Temperature"})
luffy = luffy.rename(columns={"Data Medicao": "Date"})
luffy.drop("2016-03-10", inplace = True)
luffy.drop("2017-11-15", inplace = True)
luffy.drop(luffy.index[0:17], inplace = True)

df = luffy[:'2020-01-01']
df['Pinheiros'] = bikers['Pinheiros'].values
#print (df[df.Temperature < 20])

'''
Variables to configure the day of the week and number of datas
'''
print ("\nPlease enter the dates (from 2016-01-18 and 2020-01-01):")
startDate = input("Initial date (YYYY-mm-dd): ")
endDate = input("Final date: (YYYY-mm-dd): ")
data = df[startDate:endDate]


print ("\n\nPlease enter the cut-off temperature cT in C [values will be <= cT ]:")
cT = float(input("cT (0 for no filtering): "))
if(cT > 0):data = data[data.Temperature < cT]

print ("\n\nEnter the number of day of the week:")
print ("   * 1 -> Mon")
print ("   * 2 -> Tue")
print ("   * 3 -> Wed")
print ("   * 4 -> Thu")
print ("   * 5 -> Fri")
print ("   * 6 -> Sat")
print ("   * 7 -> Sun")
print ("   * 0 -> Everyday")

dayWeek = int(input("Day: "))
dayWeekStr = ["Everyday","Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
'''
print ("\n\nEnter the type of graph:")


print ("   * 1 -> Line     (Number of Cyclist and Temperature vs Date)")
print ("   * 2 -> Bar      (Number of Cyclists vs Temperature)")
print ("   * 3 -> Scatter  (Number of Cyclists vs Temperature)\n")
typeGraph = int(input("Type: "))
'''

dayWeekInt = 0
dayWeekInit = findDay(startDate)
for i in dayWeekStr:
	if(dayWeekInit == i):
		break;
	else:
		dayWeekInt += 1

checkDayWeek = dayWeekInt - dayWeek
if (checkDayWeek > 0):
	n = checkDayWeek-1
else:
	n = 6+checkDayWeek


if dayWeek != 0:

	nMed = 6
	newDate = []
	newTemp = []
	newBikers = []
	m = 0

	for i in data.index:
		if(n == nMed):
			newDate.append(i)
			newTemp.append(data.iloc[m, 0])
			newBikers.append(data.iloc[m, 1])
			n = 0
		else:
			n += 1

		m += 1

	newData = pd.DataFrame()
	newData['Date'] = newDate
	newData['Temperature'] = newTemp
	newData['Pinheiros'] = newBikers
	newData.set_index('Date', inplace = True)
	print (newData)

	x = newData['Temperature']
	y = newData['Pinheiros']

	a = 0.05
	cI = 1 - a/2

	xy = x*y
	x2 = x*x
	y2 = y*y
	Sx = x.sum()
	Sy = y.sum()
	N = int(newData.size/2)

	nom = N*((xy).sum()) - (Sx*Sy)
	den = math.sqrt( (N*((x2).sum()) - (Sx*Sx)) * (N*((y2).sum()) - (Sy*Sy)) )
	r =  nom/den

	zR = 0.5*math.log( (1+r)/(1-r) )
	zA = 0.5*math.log( (1+cI)/(1-cI) )
	zL = zR - (zA)*(math.sqrt(1/(N-3)))
	zU = zR + (zA)*(math.sqrt(1/(N-3)))
	rL = (math.exp(2*zL) - 1)/(math.exp(2*zL) + 1)
	rU = (math.exp(2*zU) - 1)/(math.exp(2*zU) + 1)


	fig1, ax1 = plt.subplots()
	plt.grid(True)
	ax1.set_title(f'Daily Temperature and Daily Number of Cyclists\nevery {dayWeekStr[dayWeek]} between {startDate} and {endDate}\nSample size (n): {N}   Correlation coefficient (r): {r:.3f}   Confidence Interval (95%): [{rL:.3f}:{rU:.3f}]')
	ax1.set_xlabel("Temperature (C)")
	ax1.set_ylabel("Number of Cyclists (N)")
	ax1.tick_params(axis='x', labelrotation=45)
	ax1.scatter(newData['Temperature'].values, newData['Pinheiros'].values, label="Number of Cyclists towards Pinheiros (N)")
	ax1.plot(x, linregress(x, y)[1] + linregress(x, y)[0]*x, "g" )
	fig1.savefig(f'./scatterGraphs/Bikers-Temp_x_Date-'+dayWeekStr[dayWeek]+'-'+startDate+'-'+endDate+'-scatter.png', format='png')


	fig2, ax2 = plt.subplots()
	plt.grid(True)
	ax2.set_title(f'Daily Temperature and Daily Number of Cyclists\nevery {dayWeekStr[dayWeek]} between {startDate} and {endDate}\nSample size (n): {N}   Correlation coefficient (r): {r:.3f}   Confidence Interval (95%): [{rL:.3f}:{rU:.3f}]')
	ax2.set_xlabel("Temperature (C)")
	ax2.set_ylabel("Number of Cyclists (N)")
	ax2.tick_params(axis='x', labelrotation=45)
	ax2.bar(newData['Temperature'].values, newData['Pinheiros'].values, label="Number of Cyclists towards Pinheiros (N)")
	fig2.savefig(f'./barGraphs/Bikers-Temp_x_Date-'+dayWeekStr[dayWeek]+'-'+startDate+'-'+endDate+'-bar.png', format='png')


	fig3, ax3 = plt.subplots()
	plt.grid(True)
	ax3.set_title(f'Daily Temperature and Daily Number of Cyclists\nevery {dayWeekStr[dayWeek]} between {startDate} and {endDate}\nSample size (n): {N}   Correlation coefficient (r): {r:.3f}   Confidence Interval (95%): [{rL:.3f}:{rU:.3f}]')
	ax3.set_xlabel("Date")
	ax3.set_ylabel("Number of Cyclists (N)")
	ax3.tick_params(axis='x', labelrotation=45)
	ax3.set_xlim(data.index[0], data.index[-1])
	plot1 = ax3.plot(newData.index, newData['Pinheiros'].values, 'g', label="Number of Cyclists towards Pinheiros (N)")
	ax4 = ax3.twinx()
	ax4.set_ylabel("Temperature (C)")
	plot2 = ax4.plot(newData.index, newData['Temperature'].values, 'r', label="Average Temperature (C)")
	lns = plot1 + plot2
	labels = [l.get_label() for l in lns]
	plt.legend(lns, labels, loc=0)
	date_form = DateFormatter("%b %y")
	ax3.xaxis.set_major_formatter(date_form)
	ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
	fig3.savefig(f'./lineGraphs/Bikers-Temp_x_Date-'+dayWeekStr[dayWeek]+'-'+startDate+'-'+endDate+'-line.png', format='png')
	plt.show()

else:

	print (data)
	x = data['Temperature']
	y = data['Pinheiros']

	a = 0.05
	cI = 1 - a/2

	xy = x*y
	x2 = x*x
	y2 = y*y
	Sx = x.sum()
	Sy = y.sum()
	N = int(data.size/2)

	nom = N*((xy).sum()) - (Sx*Sy)
	den = math.sqrt( (N*((x2).sum()) - (Sx*Sx)) * (N*((y2).sum()) - (Sy*Sy)) )
	r =  nom/den

	zR = 0.5*math.log( (1+r)/(1-r) )
	zA = 0.5*math.log( (1+cI)/(1-cI) )
	zL = zR - (zA)*(math.sqrt(1/(N-3)))
	zU = zR + (zA)*(math.sqrt(1/(N-3)))
	rL = (math.exp(2*zL) - 1)/(math.exp(2*zL) + 1)
	rU = (math.exp(2*zU) - 1)/(math.exp(2*zU) + 1)

	fig1, ax1 = plt.subplots()
	plt.grid(True)
	ax1.set_title(f'Daily Temperature and Daily Number of Cyclists\neveryday between {startDate} and {endDate}\nSample size (n): {N}   Correlation coefficient (r): {r:.3f}   Confidence Interval (95%): [{rL:.3f}:{rU:.3f}]')
	ax1.set_xlabel("Temperature (C)")
	ax1.set_ylabel("Number of Cyclists (N)")
	ax1.tick_params(axis='x', labelrotation=45)
	ax1.scatter(data['Temperature'].values, data['Pinheiros'].values, label="Number of Cyclists towards Pinheiros (N)")
	ax1.plot(x, linregress(x, y)[1] + linregress(x, y)[0]*x, "g" )
	fig1.savefig(f'./scatterGraphs/Bikers-Temp_x_Date-Everyday-'+startDate+'-'+endDate+'-scatter.png', format='png')


	fig2, ax2 = plt.subplots()
	plt.grid(True)
	ax2.set_title(f'Daily Temperature and Daily Number of Cyclists\neveryday between {startDate} and {endDate}\nSample size (n): {N}   Correlation coefficient (r): {r:.3f}   Confidence Interval (95%): [{rL:.3f}:{rU:.3f}]')
	ax2.set_xlabel("Temperature (C)")
	ax2.set_ylabel("Number of Cyclists (N)")
	ax2.tick_params(axis='x', labelrotation=45)
	ax2.bar(data['Temperature'].values, data['Pinheiros'].values, label="Number of Cyclists towards Pinheiros (N)")
	fig2.savefig(f'./barGraphs/Bikers-Temp_x_Date-Everyday-'+startDate+'-'+endDate+'-bar.png', format='png')


	fig3, ax3 = plt.subplots()
	plt.grid(True)
	ax3.set_title(f'Daily Temperature and Daily Number of Cyclists\neveryday between {startDate} and {endDate}\nSample size (n): {N}   Correlation coefficient (r): {r:.3f}   Confidence Interval (95%): [{rL:.3f}:{rU:.3f}]')
	ax3.set_xlabel("Date")
	ax3.set_ylabel("Number of Cyclists (N)")
	ax3.tick_params(axis='x', labelrotation=45)
	ax3.set_xlim(data.index[0], data.index[-1])
	plot1 = ax3.plot(data.index, data['Pinheiros'].values, 'g', label="Number of Cyclists towards Pinheiros (N)")
	ax4 = ax3.twinx()
	ax4.set_ylabel("Temperature (C)")
	plot2 = ax4.plot(data.index, data['Temperature'].values, 'r', label="Average Temperature (C)")
	lns = plot1 + plot2
	labels = [l.get_label() for l in lns]
	plt.legend(lns, labels, loc=0)
	date_form = DateFormatter("%b %y")
	ax3.xaxis.set_major_formatter(date_form)
	ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
	fig3.savefig(f'./lineGraphs/Bikers-Temp_x_Date-Everyday-'+startDate+'-'+endDate+'-line.png', format='png')
	plt.show()

print ("\n*************************************")
print ("Statistical points\n")
print (f"Sample size (n): {N}")
print (f"Correlation coefficient (r): {r:.3f}")
print (f"Confidence Interval (95%): [{rL:.3f}:{rU:.3f}]")
print (f"Intercept: {linregress(x, y)[1]:.3f}")
print (f"Slope: {linregress(x, y)[0]:.3f}")
print ("*************************************\n")

'''
Finishing the script
'''

sys.exit()
