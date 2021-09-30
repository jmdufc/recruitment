import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
import pandas as pd
from highlight_text import ax_text, fig_text
from adjustText import adjust_text
from mplsoccer import PyPizza

data="playerrec.csv" 
image1="DUFCcrest_small.png"

st.set_page_config(
     page_title="Player Recruitment",
     layout="wide",
     )
col1, mid, col2 = st.columns([1,0.5,20])
with col1:
    st.image(image1)
with col2:
    st.title('Dundee United - Player Recruitment')
    
password_guess = st.sidebar.text_input('Enter password')  
if password_guess != 'TannadiceStreet':
     st.stop()

st.sidebar.image(image1, use_column_width=False)

textc='#1d3557'
linec='#808080'
font='Arial'
bgcolor="#FAF9F6"#'#f1faee'
color1='#e63946'
color2='#a8dadc'
color3='#457b9d'
color4='#B2B3A9'
color5='#1d3557'
color6="#006daa"
pathcolor="#C4D5CB"
arrowedge=bgcolor
slice_colors =  [color1]*3 +[color2]*8 +[color3] *3
text_colors = ["black"] * 14
back_colors = ['white']*14
edge_colors = ["white"] * 14 
cellColors=[[bgcolor,bgcolor]]*14

player="Player"
param1='Defensive duels per 90'
param2='Defensive duels won, %'
param3='PAdj Interceptions'
param4='Received passes per 90'
param5='Passes to penalty area per 90'
param6='Passes to final third per 90'
param7='Progressive passes per 90'
param8='Accurate passes, %'
param9='Progressive runs per 90'
param10='Average pass length, m'
param11='% of passes progressive'
param12='xA per 90'
param13='xG per 90'
param14='Shots per 90'

@st.cache(allow_output_mutation=True)
def get_data(file):
    df=pd.read_csv(file).fillna(0)
    df.drop(df.columns[0], axis=1, inplace=False)
    return (df)
df=get_data(data)
df['Contract expires'] = df['Contract expires'].astype(str)
df["% of passes progressive"]=df["Progressive passes per 90"]/df["Passes per 90"]*100
df['color'] = np.where(df['Rank']>=1, color5, color1)
df=df.fillna(0)



leagues = list(df['League'].drop_duplicates())
leagues=sorted(leagues)
league_choice = st.sidebar.selectbox(
    "Select a league:", leagues, index=0)
df1=df.loc[(df['League'] == league_choice)]

positions = list(df1['Focus Position'].drop_duplicates())
positions=sorted(positions)
position_choice = st.sidebar.selectbox(
    "Select a position:", positions, index=2)
df1=df1.loc[(df1['Focus Position'] == position_choice)]

af=df.loc[(df['Focus Position'] == position_choice)]

df1_1=df1.reset_index(drop=True)
#df1_2=df1_1

df2=df1.iloc[:,9:]
metrics=df2.columns.tolist()
remove=['Birth country','Passport country','Foot','Height','Weight','On loan','League',
        'Focus Position','Rank','Status', 'color']

for ele in remove:
    metrics.remove(ele)

metric1 = st.sidebar.selectbox(
    "Choose a metric (Scatter plot):", metrics, index=0)

metric2 = st.sidebar.selectbox(
    "Choose a metric (Scatter plot):", metrics, index=1)


age_min =int(df1['Age'].min())
age_max =int(df1['Age'].max())
min_age, max_age = st.sidebar.slider(
    'Filter by age:',age_min,age_max,(age_min,age_max))

df1=df1.loc[(df1['Age'] >= min_age)]
df1=df1.loc[(df1['Age'] <= max_age)]


height_min =int(df1['Height'].min())
height_max =int(df1['Height'].max())
min_height, max_height = st.sidebar.slider(
    'Filter by height (cm):',height_min,height_max,(height_min,height_max))

df1=df1.loc[(df1['Height'] >= min_height)]
df1=df1.loc[(df1['Height'] <= max_height)].reset_index(drop=True)

st.subheader(league_choice + " - " + position_choice)

fig = plt.figure(figsize=(10,10),constrained_layout=True)
gs = fig.add_gridspec(nrows=1,ncols=1)
fig.patch.set_facecolor(bgcolor)

ax1 = fig.add_subplot(gs[0])

ax1.set_title(label=f"Scatter Plot: {metric1} vs {metric2}",x=0.5,y=1.05,size=20,color=textc,ha='center',fontweight='bold')
#ax1.text(s=f"Blue bars represent league average",x=0.05,y=0,size=10,color=textc)

var1=metric1
var2=metric2

x1 = df1[var1]
y2 = df1[var2]

mean1 = df1_1[var1].mean()
mean2 = df1_1[var2].mean()
max1 = df1_1[var1].max()
max2 = df1_1[var2].max()
min1 = df1_1[var1].min()
min2 = df1_1[var2].min()


ax1.scatter(x1, y2, color=df1.color,edgecolor=df1.color,alpha=0.5, s=100,zorder=4)

ax1.plot([min1,max1],[mean2,mean2],color=color2,lw=50,alpha=0.25)
ax1.plot([mean1,mean1],[min2,max2],color=color2,lw=50,alpha=0.25)


players=df1['Player']

# set the background color for the axes
ax1.set_facecolor(bgcolor)

# player names with their coordinate locations   

df1_=df1.loc[(df1[var1] >= mean1) | (df1[var2] >= mean2)]

text_values = df1_.loc[
    df1_["Player"].isin(players),
    [var1, var2, "Player"]
].values

# make an array of text
texts = [
    ax1.text(
        val[0], val[1], val[2], 
        size=16, color=textc, zorder=5,#rotation=45,
        fontfamily=font
    ) for val in text_values
]

# use adjust_text
adjust_text(
   texts, autoalign='x', 
    only_move={'points':'y', 'text':'xy'}, 
    force_objects=(0, 0), force_text=(0, 0), 
    force_points=(0, 0)
)

    
# add x-label and y-label
ax1.set_xlabel(
    var1, color=textc,
    fontsize=18, fontfamily=font
)
ax1.set_ylabel(
    var2,color=textc,
    fontsize=18, fontfamily=font
)

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_xlim(min1*0.9,max1*1.1)
ax1.set_ylim(min2*0.9,max2*1.1)

#TOP 5 TARGETS
df3=df.loc[(df['Rank'] >= 1) & (df['League'] == league_choice) & (df['Focus Position'] == position_choice)]
df3=df3.sort_values('Rank')

cols= ['Player','Team','Age','Contract expires','Rank','Status']

df3 = df3[cols].reset_index(drop=True)

### COMPARISON RADAR
st.sidebar.write("Choose players to compare:")

index = df1_1.index
condition = df1_1["Rank"] == 1
player_ind = index[condition]
player_ind=player_ind.tolist()

player_ind_b = ((af.League.values == league_choice) & (af["Focus Position"].values == position_choice)).argmax()
player_ind_b = [player_ind_b]

if len(player_ind)>0:
    player_ind=player_ind
else:
    player_ind=player_ind_b#[0]
player_ind = int(''.join(str(i) for i in player_ind))


condition2 = df1_1["Rank"] == 2
player_ind2 = index[condition2]
player_ind2=player_ind2.tolist()

player_ind2_b = ((af.League.values == league_choice) & (af["Focus Position"].values == position_choice)).argmax()+1
player_ind2_b = [player_ind2_b]

if len(player_ind2)>0:
    player_ind2=player_ind2
else:
    player_ind2=player_ind2_b#[1]
player_ind2 = int(''.join(str(i) for i in player_ind2))


player1 = list(af['Player'].drop_duplicates())
#player1=sorted(player1)
player1_choice = st.sidebar.selectbox(
    "Select player 1:", player1, index=player_ind)

player2_choice = st.sidebar.selectbox(
    "Select player 2:", player1, index=player_ind2)


params = [param1,param2,param3,param4,param5,param6,
                               param7,param8,param9,param10,param11,param12,param13,param14]

params2 = ['Def duels\np90','Def duels\nwon%','PAdj\nInterceptions','Received pass\np90','Passes to\npen area p90','Passes to final\n3rd p90',
                               'Prog. passes\np90','Accurate\npasses %','Prog runs\np90','Avg pass\nlength (m)',
                               "% of passes\nprogressive",'xA p90','xG p90','Shots p90']


def get_league(player,df):
    league=df.loc[df['Player'] ==player,'League']#.to_string(index=False)
    league=league.head(1).item()
    position=position_choice #df.loc[df['Player'] ==player,'Focus Position']#.to_string(index=False)
    #position=position.head(1).item()
    radar=df.loc[(df['League']==league) & (df['Focus Position']==position)]
    players_hold=radar['Player']
    num_cols = radar.select_dtypes([np.number]).columns
    radardf = radar[num_cols].rank(0,ascending=True, pct=True,method='average')*100
    radardf=pd.DataFrame(radardf, columns=[param1,param2,param3,param4,param5,param6,
                               param7,param8,param9,param10,param11,param12,param13,param14])
    radardf=pd.concat([players_hold,radardf], axis=1).reset_index(drop=True)
    return(radardf)

radar1=get_league(player1_choice,df)
radar2=get_league(player2_choice,df)


#WORK FROM HERE
    

def get_ranges(rankdf,player):
    a_values = []
    
    for x in range(len(rankdf['Player'])):
        if rankdf['Player'][x] == player:
            a_values = rankdf.iloc[x].values.tolist()
        
    a_values = a_values[1:]

    a_values=list(map(int, a_values))
    
    return(a_values)

a_values1=get_ranges(radar1,player1_choice)
a_values2=get_ranges(radar2,player2_choice)


fig1 = plt.figure(figsize=(10,18),constrained_layout=True)
gs = fig1.add_gridspec(nrows=1,ncols=2)
fig1.patch.set_facecolor(bgcolor)

baker = PyPizza(
    params=params2,                  # list of parameters
    background_color="#2D302D",     # background color
    straight_line_color="#2D302D",  # color for straight lines
    straight_line_lw=1,             # linewidth for straight lines
    last_circle_lw=0,               # linewidth of last circle
    other_circle_lw=0,              # linewidth for other circles
    inner_circle_size=10            # size of inner circle
)


# plot pizza

league1=df.loc[df['Player'] ==player1_choice,'League']#.to_string(index=False)
league1=league1.head(1).item()
ax1 = fig1.add_subplot(gs[0,0],projection='polar')
ax1.set_title(label=f"{player1_choice}\n{league1}",x=0.5,y=1.1,size=14,color=textc,ha='center',fontweight='bold')

baker.make_pizza(
    a_values1,  # list of values
    ax=ax1,                # adjust figsize according to your need
    color_blank_space="same",        # use same color to fill blank space
    slice_colors=slice_colors,       # color for individual slices
    value_colors=text_colors,        # color for the value-text
    value_bck_colors=back_colors,   # color for the blank spaces
    blank_alpha=0.2,                 # alpha for blank-space colors
    kwargs_slices=dict(
        edgecolor=edge_colors, zorder=2, linewidth=1
    ),                                     # values to be used when plotting slices
    kwargs_params=dict(
        color=textc, fontsize=10,
         va="center"
    ),                               # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=11,fontfamily=font,
         zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
        )
    )                                # values to be used when adding parameter-values
)

league2=df.loc[df['Player'] ==player2_choice,'League']#.to_string(index=False)
league2=league2.head(1).item()
ax2 = fig1.add_subplot(gs[0,1],projection='polar')
ax2.set_title(label=f"{player2_choice}\n{league2}",x=0.5,y=1.11,size=14,color=textc,ha='center',fontweight='bold')


baker.make_pizza(
    a_values2,  # list of values
    ax=ax2,                # adjust figsize according to your need
    color_blank_space="same",        # use same color to fill blank space
    slice_colors=slice_colors,       # color for individual slices
    value_colors=text_colors,        # color for the value-text
    value_bck_colors=back_colors,   # color for the blank spaces
    blank_alpha=0.2,                 # alpha for blank-space colors
    kwargs_slices=dict(
        edgecolor=edge_colors, zorder=2, linewidth=1
    ),                                     # values to be used when plotting slices
    kwargs_params=dict(
        color=textc, fontsize=10,
         va="center"
    ),                               # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=11,fontfamily=font,
         zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
        )
    )                                # values to be used when adding parameter-values
)

fig1.text(
    0.05, 0.75,s=f"Comparing {position_choice}", size=24,fontfamily=font,
    color=textc
)

fig1.text(
    0.05, 0.7,s=f"The percentile rank shows where the player ranks for each attribute compared\nto peers within his league - on a scale of 1-100.", size=16,fontfamily=font,
     color=textc
)

col1, col2 =st.columns(2)

with col1:
    st.write("Blue bars represent league average")
    st.pyplot(fig)
with col2:
    st.subheader("Top 5 Targets")
    st.write(df3)
    st.subheader("Compare players:")
    st.pyplot(fig1)

st.subheader("Full Data Set: " + league_choice + " - " + position_choice)


#player_df = list(df1['Player'].drop_duplicates())
#player_df_choice = st.sidebar.selectbox(
 #   "Select a player:", player_df, index=[0])

df1=df1.astype(str)
st.dataframe(df1,1000,2500)
