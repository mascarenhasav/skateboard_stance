'''
*************************************************************
Mascarenhas Alexandre
Experimental Design in Computer Science 2022/1 University of Tsukuba
Report 2
SKATEBOARDING STANCE AND HANDEDNESS: A BRIEF ANALYSIS OF RELATIONSHIP, PROPORTIONS AND INFLUENCES
Professor: Claus Aranha
*************************************************************
'''
#libs
from matplotlib.font_manager import FontProperties
from pandas.core.algorithms import value_counts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
from scipy.stats.distributions import ncx2
import statsmodels.stats.power as smp
import scipy.stats as st
pd.set_option('mode.chained_assignment', None)
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = 'cornsilk'
plt.rcParams['savefig.facecolor'] = 'black'
plt.rcParams['text.color'] = 'black'


share_url = "https://docs.google.com/spreadsheets/d/1A9necJ2qDoMa9k6l-eG8QP2orVHBr4P6OypA8Xy_cKA/edit#gid=2089872127"
url = share_url.replace('/edit#gid=', '/export?format=csv&gid=')
data = pd.read_csv(url)

def barChart(leftPie, rightPie, mode):
    # set width of bar
    barWidth = 0.20
    fig = plt.subplots(figsize =(12, 8))

    # set height of bar
    regular = [leftPie[0], rightPie[0]]
    goofy = [leftPie[1], rightPie[1]]

    # Set position of bar on X axis
    br1 = np.arange(len(regular))
    br2 = [x + 1.1*barWidth for x in br1]


    # Make the plot
    plt.bar(br1, regular, color ='tab:red', width = barWidth,
            edgecolor ='grey', label ='Regular')
    plt.bar(br2, goofy, color ='tab:blue', width = barWidth,
            edgecolor ='grey', label ='Goofy')

    plt.axvline((br1[1]+br2[0])/2, color='black', linestyle='--', linewidth=2)

    plt.text(br1[0]-barWidth/10, regular[0]+0.5, f"{regular[0]:.0f}", fontsize=14, weight="bold", c='black')
    plt.text(br2[0]-barWidth/10, goofy[0]+0.5, f"{goofy[0]:.0f}", fontsize=14, weight="bold", c='black')
    plt.text(br1[1]-barWidth/10, regular[1]+0.5, f"{regular[1]:.0f}", fontsize=14, weight="bold", c='black')
    plt.text(br2[1]-barWidth/10, goofy[1]+0.5, f"{goofy[1]:.0f}", fontsize=14, weight="bold", c='black')

    # Adding Xticks
    #plt.xlabel('Handedness', fontweight ='bold', fontsize = 22)
    #plt.grid(True)
    plt.grid(color='gray', linestyle='dashed', linewidth=.5)
    plt.title(f"Relation between Stance and Handedness in {mode} population", color='cornsilk', fontsize=20)
    plt.ylabel('Number of people', fontsize = 22)
    plt.xticks([r+0.55*barWidth for r in br1], ['Left-handed', 'Right-handed'], fontsize=22, color="cornsilk")
    plt.yticks(fontsize=15, color="cornsilk")

    for text in plt.legend().get_texts():
        text.set_color('black')
        text.set_fontsize(30)


    plt.savefig(f"bar_{mode}.png")
    plt.show()


#Yates’ continuity correction in values Freq
def yatesCorrection(e):
    x = e.copy()

    x.at['Left', 'Regular'] += 0.5
    x.at['Right', 'Regular'] = x.at['Total', 'Regular'] - x.at['Left', 'Regular']
    x.at['Right', 'Goofy'] = x.at['Right', 'Total'] - x.at['Right', 'Regular']
    x.at['Left', 'Goofy'] = x.at['Left', 'Total'] - x.at['Left', 'Regular']

    return x


def funcL(val):
    return f'{val:.1f}%'
    #return f'{val:.1f}%'

def funcR(val):
    #return f'{val / 100 * int(right.size/2):.1f}\n\n{val:.1f}%'
    return f'{val:.1f}%'

def pieChart(leftPie, rightPie, mode):
    # Wedge properties
    wp = { 'linewidth' : 2, 'edgecolor' : "black"}
    # Creating color parameters
    colorsL = ( "tab:red", "tab:blue")
    colorsR = ( "tab:red", "tab:blue")

    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(15, 7))

    labels = ['Regular', 'Goofy']
    wedges, texts, autotexts = ax1.pie(leftPie,
                                    autopct = funcL,
                                    shadow = True,
                                    colors = colorsL,
                                    startangle = 0,
                                    wedgeprops = wp,
                                    textprops = dict(color ="black", fontsize=28))

    wedges, texts, autotexts = ax2.pie(rightPie,
                                    autopct = funcR,
                                    shadow = True,
                                    colors = colorsR,
                                    startangle = 0,
                                    wedgeprops = wp,
                                    textprops = dict(color ="black", fontsize=28))

    # Adding legend
    '''
    ax1.legend(wedges, labels,
            title ="Stance",
            loc ="center left",
            bbox_to_anchor =(-0.17, 0.3, 0.5, 1))
    '''
    #plt.title(f"Relation between Stance and Handedness in {mode} population", color='cornsilk', fontsize=20)
    ax2.legend(wedges, labels,
            title ="Stance",
            loc ="center left",
            fontsize=30,
            bbox_to_anchor =(-0.6, -0.3, 0.5, 1))


    ax1.set_title(f"Left-handed", size=25, c="cornsilk")
    ax2.set_title("Right-handed", size=25, c="cornsilk")
    plt.tight_layout()
    plt.savefig(f"pie_{mode}.png")
    plt.show()

def chi2Eval(o, e):
    sum = 0
    for i in range(e.columns.values.size-1):
        for j in range(len(e.index)-1):
            #print(f'E{i}{j}:{valuesFreq.iloc[i][j]:.2f}     O{i}{j}:{values.iloc[i][j]}')
            diff = ( (e.iloc[i][j] - o.iloc[i][j])**2 ) / e.iloc[i][j]
            sum += diff
    return sum


def residuals(o, e):
    N = o.iloc[-1][-1]
    row = [[], []]
    for i in range(e.columns.values.size-1):
        for j in range(len(e.index)-1):
            #print(f'E{i}{j}:{e.iloc[i][j]:.2f}     O{i}{j}:{o.iloc[i][j]}')
            num = o.iloc[i][j] - e.iloc[i][j]
            den = e.iloc[i][j] * (1-(o.iloc[i][-1]/N) ) * (1-(o.iloc[-1][j]/N))
            root = np.sqrt(den)
            row[i].append(num/root)

    res = pd.DataFrame(row)
    return res


def bootstrap(dataO, dataE, n, a, func):
    bs_chi = np.empty(n)
    dataE.drop(dataE.index[-1], inplace=True)
    dataE.set_index([[0, 1]], inplace=True)
    print(dataE)
    print(dataE.iloc[:, 0])
    print(dataE.iloc[:, 1])
    for i in range(n):
        bs_sample1 = np.random.choice(dataE.iloc[:, 0], 2)
        #print(bs_sample1)
        bs_sample2 = np.random.choice(dataE.iloc[:, 1], 2)
        #print(bs_sample1)
        ox = {'Regular':bs_sample1, 'Goofy':bs_sample2}
        bs_sample = pd.DataFrame(ox)
        bs_sample.set_index([[0, 1]], inplace=True)
        #print(bs_sample)
        bs_chi[i] = func(dataO, bs_sample)
    return bs_chi

def chi_squared(o, e):
    n = o.iloc[2][2]
    chisq_value = chi2Eval(o, e)
    p_value = chi2.sf(chisq_value,1)
    res = residuals(o, e)
    phi = np.sqrt(chisq_value/n)
    chisq_critic = chi2.ppf(1-.5, df=1)
    power = chi2.cdf(x=chisq_value, df=1)
    return chisq_value, p_value, phi, power, res

def risk_ratio(values):
    a = values.iloc[0][0]
    b = values.iloc[0][1]
    c = values.iloc[1][0]
    d = values.iloc[1][1]
    n = values.iloc[2][2]

    RR = (a/(a+b)) / (c/(c+d))
    SE_RR = np.sqrt((1/a) - (1/(a+b)) + (1/c) - (1/(c+d)))
    lb_RR = np.exp(np.log(RR) - 1.96*(SE_RR))
    ub_RR = np.exp(np.log(RR) + 1.96*(SE_RR))

    return RR, lb_RR, ub_RR

def odds_ratio(values):
    a = values.iloc[0][0]
    b = values.iloc[0][1]
    c = values.iloc[1][0]
    d = values.iloc[1][1]
    n = values.iloc[2][2]

    OR = ( (a/c)/(b/d) )
    SE_OR = np.sqrt((1/a) + (1/b) + (1/c) + (1/d))
    lb_OR = np.exp(np.log(OR) - 1.96*(SE_OR))
    ub_OR = np.exp(np.log(OR) + 1.96*(SE_OR))

    return OR, lb_OR, ub_OR

def a_r(values, RR):
    a = values.iloc[0][0]
    b = values.iloc[0][1]
    c = values.iloc[1][0]
    d = values.iloc[1][1]
    n = values.iloc[2][2]

    pE = a+b/n
    AR = (pE*(RR-1)) / (1 + pE*(RR-1))
    u = 1.96*((a+c)*(c+d)/(a*d - b*c))*( np.sqrt( (a*d*(n-c) + c*c*b) / (n*c*(a+c)*(c+d)) ) )
    lb_AR = (a*d - b*c)*np.exp(u) / (n*c + (a*d - b*c)*np.exp(u))
    ub_AR = (a*d - b*c)*np.exp(-u) / (n*c + (a*d - b*c)*np.exp(-u))

    return AR, lb_AR, ub_AR



def extractDF(df):
    leftPie = []
    rightPie = []
    right_regular = int(df[ (df['handedness'] == 'R') & (df['stance'] == 'R') ].size/2)
    right_goofy = int(df[ (df['handedness'] == 'R') & (df['stance'] == 'G') ].size/2)
    left_regular = int(df[ (df['handedness'] == 'L') & (df['stance'] == 'R') ].size/2)
    left_goofy = int(df[ (df['handedness'] == 'L') & (df['stance'] == 'G') ].size/2)
    total_regular = right_regular + left_regular
    total_goofy = right_goofy + left_goofy
    total_right = right_regular + right_goofy
    total_left = left_regular + left_goofy
    total = total_right + total_left
    left = df[(df['handedness'] == 'L')]
    left['stance'] = left['stance'].replace(['R'],0)
    left['stance'] = left['stance'].replace(['G'],1)
    right = df[(df['handedness'] == 'R')]
    right['stance'] = right['stance'].replace(['R'],0)
    right['stance'] = right['stance'].replace(['G'],1)
    values = pd.DataFrame({'Handedness':['Right', 'Left', 'Total'], 'Regular':[right_regular, left_regular, total_regular], 
                        'Goofy':[right_goofy, left_goofy, total_goofy],'Total':[total_right, total_left, total]})
    values.set_index('Handedness', inplace=True)

    valuesFreq = pd.DataFrame({'Handedness':['Right', 'Left', 'Total'], 'Regular':[total_regular*(total_right/total), total_regular*(total_left/total), total_regular],
                            'Goofy':[total_goofy*(total_right/total), total_goofy*(total_left/total), total_goofy], 'Total':[total_right, total_left, total]})
    valuesFreq.set_index('Handedness', inplace=True)
    leftPie.append( (left[(left['stance'] == 0)].size/2) )
    leftPie.append( (left[(left['stance'] == 1)].size/2) )
    rightPie.append( ((right[(right['stance'] == 0)].size/2) / int(right.size/2)) * 100)
    rightPie.append( ((right[(right['stance'] == 1)].size/2) / int(right.size/2)) * 100)
    leftBar = [left[ (left['stance'] == 0) ].size/2, left[ (left['stance'] == 1) ].size/2] 
    rightBar = [right[ (right['stance'] == 0) ].size/2, right[ (right['stance'] == 1) ].size/2] 

    return values, valuesFreq, left, right, leftPie, rightPie, leftBar, rightBar



data = pd.DataFrame.from_records(data.values)
data.columns = ["datetime", "handedness", "stance", "gender", "praticant"]
data['handedness'] = data['handedness'].replace(['Right'],'R')
data['handedness'] = data['handedness'].replace(['Left'],'L')
data['stance'] =     data['stance'].replace(['Regular'],'R')
data['stance'] =     data['stance'].replace(['Goofy'],'G')
data['gender'] =     data['gender'].replace(['Male'],'M')
data['gender'] =     data['gender'].replace(['Female'],'F')
data['praticant'] =  data['praticant'].replace(['Yes'],'Y')
data['praticant'] =  data['praticant'].replace(['No'],'N')
data.to_csv('data.csv', sep=';')
print("*************************************************************")
print("Mascarenhas Alexandre")
print("Experimental Design in Computer Science 2022/1\nUniversity of Tsukuba")
print("Report 2\n")
print("SKATEBOARDING STANCE AND HANDEDNESS: A BRIEF ANALYSIS OF \nRELATIONSHIP, PROPORTIONS AND INFLUENCES\n")
print("Professor: Claus Aranha")
print("*************************************************************\n")

# Total
df = data[['handedness', 'stance']]
values, valuesFreq, left, right, leftPie, rightPie, leftBar, rightBar = extractDF(df)
chisq_value, p_value, phi, power, res = chi_squared(values, valuesFreq)
valuesFreqYates = yatesCorrection(valuesFreq)
chisq_value_yates, p_value_yates, phi_yates, power_yates, res_yates = chi_squared(values, valuesFreqYates)
RR, lb_RR, ub_RR = risk_ratio(values)
OR, lb_OR, ub_OR = odds_ratio(values)
AR, lb_AR, ub_AR = a_r(values, RR)
print(f"----------------Total------------------")
print(values)
print(valuesFreq)
print(res)
print(f"alpha:0.05  | df = 1")
print(f'Chi² = {chisq_value:.2f} | p-value:{p_value:.2f} | phi:{phi:.2f} | power:{power:.2f}\n')
print(f"-------Yates---------")
print(valuesFreqYates)
print(res_yates)
print(f"alpha:0.05  | df = 1")
print(f'Chi² = {chisq_value_yates:.2f} | p-value:{p_value_yates:.2f} | phi:{phi_yates:.2f} | power:{power_yates:.2f}\n')
print(f"-------Ratios--------")
print(f'RR:{RR:.2f} CI(95%):[{lb_RR:.2f}, {ub_RR:.2f}]')
print(f'OR:{OR:.2f} CI(95%):[{lb_OR:.2f}, {ub_OR:.2f}]')
print(f'AR:{AR:.2f} CI(95%):[{lb_AR:.2f}, {ub_AR:.2f}]')
print(f"---------------------------------------\n")

# Male
df_M = data[ (data['gender'] == 'M') ]
df_M = df_M[['handedness', 'stance']]
values_M, valuesFreq_M, left_M, right_M, leftPie_M, rightPie_M, leftBar_M, rightBar_M = extractDF(df_M)
chisq_value_M, p_value_M, phi_M, power_M, res_M = chi_squared(values_M, valuesFreq_M)
valuesFreqYates_M = yatesCorrection(valuesFreq_M)
chisq_value_yates_M, p_value_yates_M, phi_yates_M, power_yates_M, res_yates_M = chi_squared(values_M, valuesFreqYates_M)
RR_M, lb_RR_M, ub_RR_M = risk_ratio(values_M)
OR_M, lb_OR_M, ub_OR_M = odds_ratio(values_M)
AR_M, lb_AR_M, ub_AR_M = a_r(values_M, RR_M)
print(f"----------------Male------------------")
print(values_M)
print(valuesFreq_M)
print(res_M)
print(f"alpha:0.05  | df = 1")
print(f'Chi² = {chisq_value_M:.2f} | p-value:{p_value_M:.2f} | phi:{phi_M:.2f} | power:{power_M:.2f}\n')
print(f"-------Yates---------")
print(valuesFreqYates_M)
print(res_yates_M)
print(f"alpha:0.05  | df = 1")
print(f'Chi² = {chisq_value_yates_M:.2f} | p-value:{p_value_yates_M:.2f} | phi:{phi_yates_M:.2f} | power:{power_yates_M:.2f}\n')
print(f"-------Ratios--------")
print(f'RR:{RR_M:.2f} CI(95%):[{lb_RR_M:.2f}, {ub_RR_M:.2f}]')
print(f'OR:{OR_M:.2f} CI(95%):[{lb_OR_M:.2f}, {ub_OR_M:.2f}]')
print(f'AR:{AR_M:.2f} CI(95%):[{lb_AR_M:.2f}, {ub_AR_M:.2f}]')
print(f"---------------------------------------\n")

# Female
df_F = data[ (data['gender'] == 'F') ]
df_F = df_F[['handedness', 'stance']]
values_F, valuesFreq_F, left_F, right_F, leftPie_F, rightPie_F, leftBar_F, rightBar_F= extractDF(df_F)
chisq_value_F, p_value_F, phi_F, power_F, res_F = chi_squared(values_F, valuesFreq_F)
valuesFreqYates_F = yatesCorrection(valuesFreq_F)
chisq_value_yates_F, p_value_yates_F, phi_yates_F, power_yates_F, res_yates_F = chi_squared(values_F, valuesFreqYates_F)
RR_F, lb_RR_F, ub_RR_F = risk_ratio(values_F)
OR_F, lb_OR_F, ub_OR_F = odds_ratio(values_F)
AR_F, lb_AR_F, ub_AR_F = a_r(values_F, RR_F)
print(f"---------------Female-----------------")
print(values_F)
print(valuesFreq_F)
print(res_F)
print(f"alpha:0.05  | df = 1")
print(f'Chi² = {chisq_value_F:.2f} | p-value:{p_value_F:.2f} | phi:{phi_F:.2f} | power:{power_F:.2f}\n')
print(f"-------Yates---------")
print(valuesFreqYates_F)
print(res_yates_F)
print(f"alpha:0.05  | df = 1")
print(f'Chi² = {chisq_value_yates_F:.2f} | p-value:{p_value_yates_F:.2f} | phi:{phi_yates_F:.2f} | power:{power_yates_F:.2f}\n')
print(f"-------Ratios--------")
print(f'RR:{RR_F:.2f} CI(95%):[{lb_RR_F:.2f}, {ub_RR_F:.2f}]')
print(f'OR:{OR_F:.2f} CI(95%):[{lb_OR_F:.2f}, {ub_OR_F:.2f}]')
print(f'AR:{AR_F:.2f} CI(95%):[{lb_AR_F:.2f}, {ub_AR_F:.2f}]')
print(f"---------------------------------------\n")

barChart(leftBar, rightBar, 'ALL')
pieChart(leftPie, rightPie, 'ALL')
print()
barChart(leftBar_M, rightBar_M, 'MALE')
pieChart(leftPie_M, rightPie_M, 'MALE')
print()
barChart(leftBar_F, rightBar_F, 'FEMALE')
pieChart(leftPie_F, rightPie_F, 'FEMALE')
