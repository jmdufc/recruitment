
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
import pandas as pd
from highlight_text import ax_text, fig_text
from adjustText import adjust_text
from mplsoccer import PyPizza

    
image1="DUFCcrest_small.png"

st.set_page_config(
     page_title="Player Recruitment",
     layout="wide",
     )
col1, mid, col2 = st.columns([1,0.5,20])
with col1:
    st.image(image1)
with col2:
    st.title('Dundee United - Player Recruitment - Manual upload')
    
    
#st.title(f"Dundee United - Player Recruitment")
#st.image(image1)
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


data = st.sidebar.file_uploader("Upload player comparison file")
if data is not None:
    df = pd.read_excel(data,engine='openpyxl')
else:
    st.write("You must upload a file")
    st.stop()


df['Contract expires'] = df['Contract expires'].astype(str)
df["% of passes progressive"]=df["Progressive passes per 90"]/df["Passes per 90"]*100
df=df.fillna(0)

df2=df.iloc[:,9:]
metrics=df2.columns.tolist()
remove=['Birth country','Passport country','Foot','Height','Weight','On loan']

for ele in remove:
    metrics.remove(ele)

metric1 = st.sidebar.selectbox(
    "Choose a metric (Scatter plot):", metrics, index=0)

metric2 = st.sidebar.selectbox(
    "Choose a metric (Scatter plot):", metrics, index=1)


age_min =int(df['Age'].min())
age_max =int(df['Age'].max())
min_age, max_age = st.sidebar.slider(
    'Filter by age:',age_min,age_max,(age_min,age_max))

df=df.loc[(df['Age'] >= min_age)]
df=df.loc[(df['Age'] <= max_age)]


height_min =int(df['Height'].min())
height_max =int(df['Height'].max())
min_height, max_height = st.sidebar.slider(
    'Filter by height (cm):',height_min,height_max,(height_min,height_max))

df=df.loc[(df['Height'] >= min_height)]
df=df.loc[(df['Height'] <= max_height)].reset_index(drop=True)

st.subheader("")

fig = plt.figure(figsize=(10,10),constrained_layout=True)
gs = fig.add_gridspec(nrows=1,ncols=1)
fig.patch.set_facecolor(bgcolor)

ax1 = fig.add_subplot(gs[0])

ax1.set_title(label=f"Scatter Plot: {metric1} vs {metric2}",x=0.5,y=1.05,size=20,color=textc,ha='center',fontweight='bold')
#ax1.text(s=f"Blue bars represent league average",x=0.05,y=0,size=10,color=textc)

var1=metric1
var2=metric2

x1 = df[var1]
y2 = df[var2]

mean1 = df[var1].mean()
mean2 = df[var2].mean()
max1 = df[var1].max()
max2 = df[var2].max()
min1 = df[var1].min()
min2 = df[var2].min()


ax1.scatter(x1, y2, color=color1,edgecolor=color3,alpha=0.5, s=100,zorder=4)

ax1.plot([min1,max1],[mean2,mean2],color=color2,lw=50,alpha=0.25)
ax1.plot([mean1,mean1],[min2,max2],color=color2,lw=50,alpha=0.25)


players=df['Player']

# set the background color for the axes
ax1.set_facecolor(bgcolor)

# player names with their coordinate locations   

df1_=df.loc[(df[var1] >= mean1) | (df[var2] >= mean2)]

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


### COMPARISON RADAR
st.sidebar.write("Choose players to compare:")



player1 = list(df['Player'].drop_duplicates())

player1_choice = st.sidebar.selectbox(
    "Select player 1:", player1, index=0)

player2_choice = st.sidebar.selectbox(
    "Select player 2:", player1, index=1)


params = [param1,param2,param3,param4,param5,param6,
                               param7,param8,param9,param10,param11,param12,param13,param14]

params2 = ['Def duels\np90','Def duels\nwon%','PAdj\nInterceptions','Received pass\np90','Passes to\npen area p90','Passes to final\n3rd p90',
                               'Prog. passes\np90','Accurate\npasses %','Prog runs\np90','Avg pass\nlength (m)',
                               "% of passes\nprogressive",'xA p90','xG p90','Shots p90']

def get_percentile(df):
    radar=df
    players_hold=radar['Player']
    num_cols = radar.select_dtypes([np.number]).columns
    radardf = radar[num_cols].rank(0,ascending=True, pct=True,method='average')*100
    radardf=pd.DataFrame(radardf, columns=[param1,param2,param3,param4,param5,param6,
                               param7,param8,param9,param10,param11,param12,param13,param14])
    radardf=pd.concat([players_hold,radardf], axis=1).reset_index(drop=True)
    return(radardf)

radar1=get_percentile(df)


def get_ranges(df,player):
    a_values = []
    
    for x in range(len(df['Player'])):
        if df['Player'][x] == player:
            a_values = df.iloc[x].values.tolist()
        
    a_values = a_values[1:]

    a_values=list(map(int, a_values))
    
    return(a_values)

a_values1=get_ranges(radar1,player1_choice)
a_values2=get_ranges(radar1,player2_choice)


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


ax1 = fig1.add_subplot(gs[0,0],projection='polar')
ax1.set_title(label=f"{player1_choice}",x=0.5,y=1.1,size=14,color=textc,ha='center',fontweight='bold')

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

ax2 = fig1.add_subplot(gs[0,1],projection='polar')
ax2.set_title(label=f"{player2_choice}",x=0.5,y=1.11,size=14,color=textc,ha='center',fontweight='bold')


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
    0.05, 0.75,s=f"Comparing players", size=24,fontfamily=font,
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
    st.subheader("Compare players:")
    st.pyplot(fig1)

st.subheader("Full Data Set: ")


#player_df = list(df1['Player'].drop_duplicates())
#player_df_choice = st.sidebar.selectbox(
 #   "Select a player:", player_df, index=[0])

df=df.astype(str)
st.dataframe(df,1000,2500)
