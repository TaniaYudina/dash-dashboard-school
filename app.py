import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import dash_table

deals= pd.read_excel('data/Deals.xlsx', dtype={'Id': 'Int64', 'CONTACTID':'Int64'})
cnts= pd.read_excel('data/Contacts.xlsx', dtype={'Id': 'Int64'})
calls= pd.read_excel('data/Calls.xlsx', dtype={'Id': 'Int64', 'CONTACTID': 'Int64'})
acts = pd.read_excel('data/Activity.xlsx')
source = pd.read_excel('data/Source.xlsx')
campaign = pd.read_excel('data/Campaign.xlsx')
spend = pd.read_excel('data/Spends.xlsx')
city = pd.read_excel('data/Geos.xlsx')
payment = pd.read_excel('data/Payment.xlsx', dtype={'Id': 'Int64'})
df_course_paid = payment[payment['Category'] != 'New']

#ABOUT block

def count_students():
  var = df_course_paid['Id'].count()
  return var
    
def count_country():
  var = city['Country'].nunique()
  return var

def cart():
  country_counts = city['Country'].value_counts()
  countries_more_than_one = country_counts[country_counts > 1].index
  city_more_one = city[city['Country'].isin(countries_more_than_one)]
  fig = px.scatter_geo(city_more_one , lat='Latitude', lon='Longitude', hover_name='City', projection='natural earth',
                      color='Country', color_discrete_sequence=px.colors.qualitative.Vivid, height=500)
  fig.update_geos( visible=False, resolution=50,  showcountries=True,  showsubunits=True,  lataxis_range=[35, 70],   lonaxis_range=[-25, 45]  ) 
  return fig

def paid_course():
  df = df_course_paid.groupby(['Product','Education Type', 'Tariff Plan', 'Payment Type']).size().reset_index(name='Count')
  fig = px.sunburst(df, path=['Product','Education Type', 'Tariff Plan', 'Payment Type'], values='Count',  height=520)
  fig.update_traces(textinfo="label+percent entry")
  return fig

def country_category():
  df = city.groupby('Country')['City Count'].sum().reset_index(name='Count')
  df = df.sort_values('Count', ascending=False).head(5)
  df['Percentage'] = df['Count'] / df['Count'].sum() * 100
  df = df.sort_values(by='Count', ascending=True)
  fig = px.bar(df, x='Count', y='Country', orientation='h', text_auto=True,  hover_data={'Percentage': ':.2f%'})
  fig.update_layout( xaxis_title=None, yaxis_title=None, height=240, bargap=0.2, title={'text': 'Countries', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 14}} ) 
  fig.update_traces(insidetextfont=dict(size=12, color='white'))
  fig.update_traces(hovertemplate='%{y} <br> %{x} <br> %{customdata[0]:.2f}%')
  return fig


def course_category():
  df = df_course_paid['Tariff Plan'].value_counts().reset_index(name="Count")
  df['Percentage'] = df['Count'] / df['Count'].sum() * 100
  df = df.sort_values(by='Count', ascending=True)
  fig = px.bar(df, x='Count', y='Tariff Plan', orientation='h', text_auto=True,  hover_data={'Percentage': ':.2f%'})
  fig.update_layout( xaxis_title=None, yaxis_title=None, height=240, bargap=0.2,  title={'text': 'Tariff Plan', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 14}} )
  fig.update_traces(insidetextfont=dict(size=12, color='white'))
  fig.update_traces(hovertemplate='%{y} <br> %{x} <br> %{customdata[0]:.2f}%')
  return fig

def payment_category():
  df = df_course_paid['Payment Type'].value_counts().reset_index(name="Count")
  df['Percentage'] = df['Count'] / df['Count'].sum() * 100
  df = df.sort_values(by='Count', ascending=True)
  fig = px.bar(df, x='Count', y='Payment Type', orientation='h', text_auto=True,  hover_data={'Percentage': ':.2f%'})
  fig.update_layout( xaxis_title=None, yaxis_title=None, height=240, bargap=0.2, title={'text': 'Payment Type', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 14}} )              
  fig.update_traces(insidetextfont=dict(size=12, color='white'))
  fig.update_traces(hovertemplate='%{y} <br> %{x} <br> %{customdata[0]:.2f}%')
  return fig



#OPERATORS block

def operator_load(table_name):
  if table_name != 'Calls Duration':
    if table_name == 'All Activities':
      dd = acts.groupby(['Year-Month', 'Owner Name']).size().reset_index(name='Count')
    else:
      dd = acts[acts['Category'] == table_name].groupby(['Year-Month', 'Owner Name']).size().reset_index(name='Count')
  else:
      dd = calls[calls['Call Duration'] > 0].groupby(['Year-Month', 'Owner Name'])['Call Duration'].mean().round(0).reset_index(name='Count')

  pivot_dd = dd.pivot(index='Owner Name', columns='Year-Month', values='Count').sort_values('Owner Name', ascending=True)
  pivot_dd.columns = pivot_dd.columns.astype(str)
  fig = px.imshow(pivot_dd, text_auto=True, color_continuous_scale=px.colors.sequential.Blues )
  fig.update_layout(title=None, xaxis_title=None, yaxis_title=None, plot_bgcolor='white',)
  fig.update_xaxes(side="top")
  #fig.update_xaxes( dtick="M1",  tickformat="%b %Y",   tickmode='linear')
  return fig

def act_category_time(df, table_name):
  global acts, cnts,  deals, calls
  if table_name != 'Calls Duration':
    res_by_tab = df.groupby('Year-Month')['Category'].value_counts().reset_index(name='Count')
  else:
    res_by_tab = calls.groupby('Year-Month')['Duration Category'].value_counts().reset_index(name='Count')
    res_by_tab = res_by_tab.rename(columns={'Duration Category': 'Category'})
  # Получаем уникальные категории
  unique_categories = res_by_tab['Category'].unique()
  # Генерируем цветовую карту
  color_map = generate_color_map(sorted(unique_categories))

  res_by_tab['Percent'] = res_by_tab.groupby('Year-Month')['Count'].transform(lambda x: 100 * x / x.sum())
  # Заменяем названия категорий на Mapped Category
  res_by_tab['Mapped Category'] = res_by_tab['Category'].map(color_map)
  fig = px.bar(res_by_tab, x="Year-Month", y="Count", color="Category", color_discrete_map=color_map, hover_data={'Percent': ':.2f'})
  fig.update_layout(title=None, xaxis_title=None, yaxis_title=None, legend=dict(title=None, orientation="h",  yanchor="bottom",  y=1.02,  xanchor="right",  x=1))
  #fig.update_xaxes( dtick="M1",  tickformat="%b %Y",   tickmode='linear')
  return fig

def operator_load_category(df, table_name):
  global acts, cnts, calls, deals
  if table_name != 'Calls Duration':
    df = df.groupby('Category')['Owner Name'].value_counts().reset_index(name='Count')
  else:
    df = calls.groupby('Duration Category')['Owner Name'].value_counts().reset_index(name='Count')
    df = df.rename(columns={'Duration Category': 'Category'})
  # Получаем уникальные категории
  unique_categories = df['Category'].unique()
  # Генерируем цветовую карту
  color_map = generate_color_map(sorted(unique_categories))
  df['Percent'] = df.groupby('Owner Name')['Count'].transform(lambda x: 100 * x / x.sum())
  df = df.sort_values('Owner Name', ascending=False)
  fig = px.bar(df, y='Owner Name', x='Count',  color='Category', color_discrete_map=color_map, title=None,  orientation='h',  hover_data={'Percent': ':.2f'}    )
  fig.update_layout( title=None, xaxis_title=None, yaxis_title=None,  legend=dict(title=None, orientation="h",  yanchor="bottom",  y=1.04,  xanchor="right",  x=1))
  fig.update_xaxes(side="top")
  return fig

def avg_act_time(table_name):

  if table_name != 'Calls Duration':
    dd = acts.copy()
    dd.set_index('Date', inplace=True)
    if table_name == 'All Activities':
      dd = dd
    else:
      dd = dd[dd['Category'] == table_name]

    # Считаем количество записей по каждому владельцу по дням
    daily_counts = dd.groupby(['Owner Name']).resample('D').size().reset_index(name='Count')
    daily_average = daily_counts.groupby('Date')['Count'].mean().reset_index(name='Average Count')

    # Считаем количество записей по каждому владельцу по неделям
    weekly_counts = dd.groupby(['Owner Name']).resample('W').size().reset_index(name='Count')
    weekly_average = weekly_counts.groupby('Date')['Count'].mean().reset_index(name='Average Count')

    # Считаем количество записей по каждому владельцу по месяцам
    monthly_counts = dd.groupby(['Owner Name']).resample('ME').size().reset_index(name='Count')
    monthly_average = monthly_counts.groupby('Date')['Count'].mean().reset_index(name='Average Count')
  if table_name == 'Calls Duration':
    dd = calls.copy()
    dd = dd.rename(columns={'Call Start Time': 'Date'})
    dd.set_index('Date', inplace=True)               
    if 'Call Duration' in dd.columns and not dd['Call Duration'].isnull().all():
        daily_counts = dd.groupby(['Owner Name']).resample('D')['Call Duration'].mean().reset_index(name='Count')
        daily_average = daily_counts.groupby('Date')['Count'].mean().reset_index(name='Average Count')
        
        # Считаем количество записей по каждому владельцу по неделям
        weekly_counts = dd.groupby(['Owner Name']).resample('W')['Call Duration'].mean().reset_index(name='Count')
        weekly_average = weekly_counts.groupby('Date')['Count'].mean().reset_index(name='Average Count')
        
        # Считаем количество записей по каждому владельцу по месяцам
        monthly_counts = dd.groupby(['Owner Name']).resample('ME')['Call Duration'].mean().reset_index(name='Count')
        monthly_average = monthly_counts.groupby('Date')['Count'].mean().reset_index(name='Average Count')

  fig = make_subplots()
    # Добавляем линии для каждого периода (по дням скрыто по умолчанию)
  fig.add_trace(go.Scatter(x=daily_average['Date'], y=daily_average['Average Count'],  hovertemplate='Date: %{x}<br>Average Count: %{y:.1f}', name='daily',
                          mode='lines', visible=True))

  # Среднее по неделям будет видно по умолчанию
  fig.add_trace(go.Scatter(x=weekly_average['Date'], y=weekly_average['Average Count'], hovertemplate='Date: %{x}<br>Average Count: %{y:.1f}', name='weekly',
                          mode='lines', visible=False))

  # Среднее по месяцам скрыто по умолчанию
  fig.add_trace(go.Scatter(x=monthly_average['Date'], y=monthly_average['Average Count'], hovertemplate='Date: %{x}<br>Average Count: %{y:.1f}', name='monthly',
                          mode='lines', visible=False))

  # Обновление видимости линий в зависимости от выбранной кнопки
  buttons = [
      dict(method='update',
          label='by days',
          args=[{'visible': [True, False, False]}]),  # Показывать только среднее по дням
      dict(method='update',
          label='by weeks',
          args=[{'visible': [False, True, False]}]),  # Показывать только среднее по неделям
      dict(method='update',
          label='by months',
          args=[{'visible': [False, False, True]}]),  # Показывать только среднее по месяцам

  ]

  # Добавление кнопок и отключение легенды
  fig.update_layout(
      title=None, xaxis_title=None, yaxis_title=None, margin=dict(t=0, l=0, r=0, b=0), 
      updatemenus=[dict(type='dropdown', buttons=buttons, direction='down', showactive=True,  yanchor="bottom",  y=1.02,  xanchor="left",  x=0)], )
  return fig

def generate_color_map(categories):
    # Генерируем уникальные цвета для каждой категории
    colors = px.colors.diverging.Picnic#px.colors.qualitative.Set1  

    # Если количество категорий больше, чем доступных цветов, дублируем цвета
    color_map = {category: colors[i % len(colors)] for i, category in enumerate(categories)}

    return color_map

def create_category_act(df, table_name):
    global acts, cnts, calls, deals

    if table_name != 'Calls Duration':
      if table_name == 'Contacts':
        df = df.groupby(['Category', 'Description']).size().reset_index(name='Count')
      else:
        df = df.groupby('Category').size().reset_index(name='Count')
    else:
      df = calls.groupby('Duration Category').size().reset_index(name='Count')
      df = df.rename(columns={'Duration Category': 'Category'})


    # Получаем уникальные категории
    unique_categories = df['Category'].unique()
    # Генерируем цветовую карту
    color_map = generate_color_map(sorted(unique_categories))

    # Расчет процента для каждой категории
    df['Percent'] = (df['Count'] / df['Count'].sum()) * 100

    if table_name == 'Contacts':
        # Если столбец 'Description' существует, используем его
        df['Label'] = df['Category'] + '<br>' + df['Count'].astype(str) + ' (' + df['Percent'].round(2).astype(str) + '%)' + '<br>' + df['Description']
    else:
        # Если столбца нет, создаем метку без описания
        df['Label'] = df['Category'] + '<br>' + df['Count'].astype(str) + ' (' + df['Percent'].round(2).astype(str) + '%)'

    # Построение простой древовидной диаграммы (квадратной), используя категорию и количество
    fig = px.treemap(df, path=['Category'], values='Count',  color='Category', color_discrete_map=color_map)

    # Обновление графика: добавление текста в центр квадрата, увеличение шрифта и отключение данных при наведении
    fig.update_traces(
        root_color="lightgrey",
        text=df['Label'],  # Передаем текст с описаниями и данными
        texttemplate='%{text}',  # Используем переданный текст
        textposition="middle center",  # Размещение текста по центру
        hoverinfo='none',  # Отключение данных при наведении
        textfont_size=12  # Увеличиваем шрифт (в два раза, исходя из стандартного значения ~12)
    )

    # Убираем корневую группу 'All' (уменьшаем отступы)
    fig.update_layout(title='Category', title_x=0.5, margin=dict(t=0, l=0, r=0, b=0) )
    return fig
    

#MARKETING block

def sourses_graph(value_vars):

    df = source[['Source', 'CTR', 'Convers UA', 'Convers C1', 'CPC', 'CAC', 'CPA', 'ARPU', 'CRR', 'MER']].round(2)

    sources_to_exclude = ['CRM', 'Offline', 'Organic', 'Partnership']
    # Исключаем строки, где значения в колонке 'Source' совпадают с указанными
    df = df[~df['Source'].isin(sources_to_exclude)]

    # Преобразуем DataFrame в длинный формат
    df_melted = df.melt(id_vars='Source', value_vars=value_vars, var_name='Metric', value_name='Value')

    # Создаем гистограмму
    fig = px.histogram(df_melted, x='Source', y='Value', color='Metric', barmode='group', height=400, color_discrete_sequence=px.colors.diverging.Picnic)

    # Если выбранная группа group3, добавляем логарифмическую шкалу
    if value_vars == ['ARPU', 'CRR', 'MER']:
        fig.update_yaxes(type='log')
    fig.update_layout(xaxis_title=None, yaxis_title=None, legend=dict(title=None, orientation="h",  yanchor="bottom",  y=1.04,  xanchor="right",  x=1))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig
    
def spend_by_month():
  sources_to_exclude = ['CRM', 'Offline', 'Organic', 'Partnership']
  spend_filter = spend[~spend['Source'].isin(sources_to_exclude)]
  spends_by_month = spend_filter.groupby(['Year-Month', 'Source'])['Spend'].sum().reset_index(name='Sum_spend')
  pivot_data = spends_by_month.pivot(index='Source', columns='Year-Month', values='Sum_spend')
  fig = px.imshow(pivot_data, text_auto=True, color_continuous_scale=px.colors.sequential.Blues )
  fig.update_layout(title='Spending by Source and Month', title_x=0.5, xaxis_title=None, yaxis_title=None, plot_bgcolor='white')
  return fig
    
def pie_source():
  pie_source = spend.groupby('Source')['Campaign'].nunique().reset_index()
  pie_source = pie_source[pie_source['Campaign'] > 0]
  aquamarine_colors = ['#00BFFF', '#40E0D0', '#48D1CC', '#20B2AA', '#008B8B']

  fig = go.Figure(data=[go.Pie(labels=pie_source['Source'], values=pie_source['Campaign'], hole=0.5, marker=dict(colors=aquamarine_colors) )])
  fig.update_layout(
      title_text="Campaigns by Source", title_x=0.44,
      annotations=[dict(text='Sources', x=0.5, y=0.5, font_size=20, showarrow=False)],

  )
  return fig



#DEALS block

#activity of advertising companies
def advert_activity():
    campaign_by_month = spend.groupby('Year-Month')['Campaign'].count()
    fig = px.line(campaign_by_month, x=campaign_by_month.index,   y=campaign_by_month.values)
    fig.update_layout(  xaxis_title=None,       yaxis_title=None,  title=dict(text='Activity of Advertising Campanies', x=0.5, y=0.9))
    return fig
#lead generation
def lead_generation():
    leads_by_month = payment.groupby('Year-Month').size().reset_index(name='Count')
    fig = px.line(leads_by_month, x='Year-Month',   y='Count')
    fig.update_layout(  xaxis_title=None,       yaxis_title=None,  title=dict(text='Lead Generation', x=0.5, y=0.9) )
    return fig
#successful deals
def success_deals():
    study_by_month = payment[payment['Category'] != 'New'].groupby(['Product', 'Year-Month']).size().reset_index(name='Count')
    fig = px.line(study_by_month,    x='Year-Month',    y='Count',    color='Product',     
                  labels={'Count': 'Number of Students', 'Year-Month': 'Month'},  line_shape='linear', 
                  color_discrete_sequence=['royalblue', 'deepskyblue', 'blue'])      
    fig.update_layout(
        title=dict(text='Successful Deals', x=0.5, y=0.96),  xaxis_title=None,  yaxis_title=None  ,
        legend=dict(x=0.7, y=0.98))
    return fig

#Closed Deals & Average Deal Duration
def problem_deals():
    monthly_deals = deals.groupby('Year-Month').agg( total_closed=('Id', 'count'), avg_duration=('Deals Duration', 'mean') ).reset_index()
    trace1 = go.Bar( x=monthly_deals['Year-Month'].astype(str),  y=monthly_deals['total_closed'],  name='Closed Deals',  marker_color='royalblue',
        yaxis='y1')
    trace2 = go.Scatter( x=monthly_deals['Year-Month'].astype(str), y=monthly_deals['avg_duration'], name='Average Deal Duration', mode='lines+markers',
        marker_color='skyblue', yaxis='y2' )
    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
        title=dict(text='Closed Deals & Average Deal Duration', x=0.5, y=0.96),
        xaxis=dict(title=None),
        yaxis=dict(
            title='Closed Deals',
            titlefont=dict(color='royalblue'),
            tickfont=dict(color='royalblue'),
            side='left'
        ),
        yaxis2=dict(
            title='Average Deal Duration (days)',
            titlefont=dict(color='deepskyblue'),
            tickfont=dict(color='deepskyblue'),
            overlaying='y',
            side='right'
        ),
        showlegend=False     
    )

    return fig

#by product
def products():
    products = payment['Product'].dropna().unique().tolist()
    
    product_data = {}
    for product in products:
        dd = payment[payment['Product'] == product].groupby(['Year-Month', 'Months of study'])['Id'].count().reset_index(name='Count')
        pivot_dd = dd.pivot(index='Year-Month', columns='Months of study', values='Count').fillna(0)
        product_data[product] = pivot_dd
    fig = go.Figure()
    for product in products:
        fig.add_trace(go.Heatmap(
            z=product_data[product].values,
            x=product_data[product].columns,
            y=product_data[product].index,
            visible=(product == products[0]),  
            colorscale= 'Portland',
            text=product_data[product].values,
            texttemplate='%{text:.0f}',  
            name=product
        ))
    buttons = [
        dict(label=product,
             method="update",
             args=[{"visible": [product == p for p in products]},  
                   {"title": f"Number of Students in '{product}'"}])  
        for product in products
    ]
    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 0.01,
            "xanchor": "left",
            "y": 1.2,
            "yanchor": "top"
        }],
        title=dict(text=f"Number of Students in '{products[0]}'", x=0.5, y=0.9),      
        xaxis_title="Months of study",
        yaxis_title=None,
        yaxis=dict(autorange='reversed')  
    )
    
    return fig

def funnel():    
    funnel_data_total = pd.DataFrame({
    'Leads': [payment['Category'].count()], 
    'First Month Paid': [payment[payment['Category'] != 'New']['Category'].count()],  
    'Students': [payment[payment['Months of study'] > 0]['Category'].count()], 
    'Finished Course': [payment[payment['Months of study'] == payment['Course duration']]['Category'].count()]  
    })  
    funnel_data_total = funnel_data_total.melt(
        value_vars=['Leads', 'First Month Paid', 'Students', 'Finished Course'],
        var_name="Stage", value_name="Count")
    
    funnel_data_total['Previous Count'] = funnel_data_total['Count'].shift(1).fillna(0)
    funnel_data_total['Conversion'] = (funnel_data_total['Count'] / funnel_data_total['Previous Count'] * 100).fillna(0)
    
    total_leads = funnel_data_total.loc[funnel_data_total['Stage'] == 'Leads', 'Count'].values[0]
    funnel_data_total['Percentage'] = (funnel_data_total['Count'] / total_leads * 100).round(2)

    fig = go.Figure(go.Funnel(
            y=funnel_data_total['Stage'],
            x=funnel_data_total['Count'],
            name="Quantity",
            marker=dict(color='royalblue')
        ))


    fig.update_layout(
        title_text=None,
        title_x=0.5,
        width=1500,
        showlegend=False
    )   
    return fig

def pie_amount():      
    amount_first = payment[payment['Category'] != 'New']['Month Priсe'].sum().astype('int')
    amount_others =  (payment[payment['Category'] != 'New']['Paid Amount'].sum() - payment[payment['Category'] != 'New']['Month Priсe'].sum()).astype('int')
    pie_source = pd.DataFrame({
        'Category': ['First Month Paid', 'Subsequent Months Paid'],
        'Amount': [amount_first, amount_others]
    })    
    aquamarine_colors = ['deepskyblue', 'royalblue']
    fig = go.Figure(data=[go.Pie(labels=pie_source['Category'], values=pie_source['Amount'], hole=0.4, marker=dict(colors=aquamarine_colors))])
    fig.update_layout(
        title_text="Distribution of Payment Amounts", 
        title_x=0.4,        
    )    
    return fig

def unit():
    unit = payment.groupby(['Product', 'Tariff Plan'])['Id'].nunique().reset_index(name='UA')
    index_unit = unit.set_index(['Product', 'Tariff Plan']).index
    unit['B'] = payment[payment['Category'] != 'New'].groupby(['Product', 'Tariff Plan']).size().reindex(index_unit, fill_value=0).astype(int).values
    unit['Students'] = payment[payment['Months of study'] > 0].groupby(['Product', 'Tariff Plan'])['Id'].count().reindex(index_unit, fill_value=0).values
    unit['Convers Study'] = (unit['Students'] / unit['B'] * 100).round(1)
    return unit
    
def plot_unit_scatter_shifted():
    unit_df = unit()
    fig = go.Figure()

    # Добавляем кружки для значений 'B' с небольшим смещением
    fig.add_trace(go.Scatter(
        x=unit_df['Product'] + " ",  # Смещаем по горизонтали, добавляя текстовую метку
        y=unit_df['Tariff Plan'],
        mode='markers',
        marker=dict(
            size=unit_df['B'],   # Размер кружков пропорционален значению 'B'
            color='royalblue',
            sizemode='area',
            sizeref=2.*max(unit_df['B'])/(100.**2),
            sizemin=4,
        ),
        name='Number of First Month Paid',
        text=unit_df['B'],  # Подсказки при наведении показывают значение 'B'
        hoverinfo='text'
    ))

    # Добавляем кружки для значений 'Convers Study' без смещения
    fig.add_trace(go.Scatter(
        x=unit_df['Product'] + "  ",  # Смещаем по горизонтали, добавляя текстовую метку
        y=unit_df['Tariff Plan'],
        mode='markers',
        marker=dict(
            size=unit_df['Convers Study'],  # Размер кружков пропорционален значению 'Convers Study'
            color='deepskyblue',
            sizemode='area',
            sizeref=2.*max(unit_df['Convers Study'])/(100.**2),
            sizemin=4,
        ),
        name='Student Retention, %',
        text=unit_df['Convers Study'],  # Подсказки при наведении показывают значение 'Convers Study'
        hoverinfo='text'
    ))

    # Настройка оформления графика
    fig.update_layout(
        title=dict(text="'First Month Paid' and 'Student Retention' by Product and Tariff Plan", x=0.08, y=0.92),        
        xaxis_title=None,
        yaxis_title=None,
        showlegend=True,
        legend=dict(title=None, orientation="h",  yanchor="bottom",  y=1.04,  xanchor="right",  x=1)
    )

    return fig
    
#ABOUT 
tab1_content = [
    dbc.Row([
        dbc.Col([
            html.H1('IT Online School'),
            html.P('*period 2023 - 2024', style={'font-size': '12px'}),
        ], width={'size': 4}),
        dbc.Col([
            html.Div([
                    html.P("Web Developer", style={'margin': '0px'}),
                    html.P("Digital Marketing", style={'margin': '0px'}),
                    html.P("UX/UI Design", style={'margin': '0px'}),
                ],style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'font-size': '18px'})
        ], width={'size': 2, 'offset': 1}),
        dbc.Col([
            html.Div([
                html.H2(count_students(), style={'margin': '0px'}),
                html.P("students", style={'margin': '0px', 'font-size': '16px'}),
            ],style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
        ], width={'size': 1, 'offset': 1}),
        dbc.Col([
            html.Div([
                html.H2(count_country(), style={'margin': '0'}),
                html.P("countries", style={'margin': '0 0 0 5px', 'font-size': '16px'}),
            ],style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
        ], width={'size': 1, 'offset': 1})
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=cart(), style={'margin': '0'})
        ], width={'size': 7}),
        dbc.Col([
            dcc.Graph(figure=paid_course())
        ], width={'size': 5}),
        
        ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=country_category(), style={'margin': '0'}),
        ], width={'size': 4}),
        dbc.Col([
            dcc.Graph(figure=course_category(), style={'margin': '0'}),
        ], width={'size': 4}),
        dbc.Col([
            dcc.Graph(figure=payment_category(), style={'margin': '0'})
        ], width={'size': 4}),
        ]),
    
]

#DEALS
tab2_content = [
    dbc.Row(
            html.H1('Deals'),
       ),
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=advert_activity(), style={'margin': '0px', 'height':'300px'}),
            dcc.Graph(figure=lead_generation(), style={'margin': '0px', 'height':'300px'}),            
        ], width={'size': 3}),
        dbc.Col([
            dcc.Graph(figure=success_deals(), style={'margin': '0px', 'height':'600px'}),          
        ], width={'size': 5}),
        dbc.Col([
            dcc.Graph(figure=problem_deals(), style={'margin': '0px', 'height':'600px'}),          
        ], width={'size': 4}),
        ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=products(), style={'margin': '0px', 'height':'500px'}),                     
        ], width={'size': 8}),
        dbc.Col([
            dcc.Graph(figure=funnel(), style={'margin': '0px', 'height':'500px'}),          
        ], width={'size': 4}),
        ]),
    html.Br(),   
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure= pie_amount(), style={'margin': '0px', 'height':'500px'}),                     
        ], width={'size': 4}),
        dbc.Col([
            dcc.Graph(figure=plot_unit_scatter_shifted(), style={'margin': '0px', 'height':'500px'}),          
        ], width={'size': 8}),
        ])
]


#OPERATORS
tab3_content = [
    dbc.Row(
            html.H1('Operators', style={'marginLeft': '60px'}),
       ),
    dbc.Row([
        dbc.Col([
            html.H1(id='table-name', style={'font-size': '28px', 'font-weight': 'bold', 'text-align': 'center', 'margin-top':'60px'}),
            html.Div(id='count_persons', style={'font-size': '28px', 'text-align': 'center'}),
            html.Div('persons', style={'text-align': 'center', 'font-size': '24px'}),
        ], width={'size': 2}),
        dbc.Col([
            html.H4(id="title_avg", style={'text-align': 'center', 'font-weight': 'normal', 'margin-bottom': '0px'}),
            html.Div(id='avg_act_time_content')
        ], width={'size': 9}),
        
        ]),
    html.Br(),
    html.Br(),
    dbc.Row(html.Div([html.H4("by Category", style={'text-align': 'center'})])),
    dbc.Row([
        dbc.Col([
            html.Br(),
            html.Br(),
            html.Div([html.Div(id='category_content')], style={'margin': '0'}),            
        ], width={'size': 5}),
        dbc.Col([
            html.Div([html.Div(id='act_category_time_content')], style={'margin': '0'}),
        ], width={'size': 7}),
        ]),
    html.Br(), 
    dbc.Row([
       dcc.Dropdown(
            id='table-dropdown',
            options=[
                {'label': 'All', 'value': 'acts'},
                {'label': 'Contacts', 'value': 'cnts'},
                {'label': 'Calls', 'value': 'calls'},
                {'label': 'Deals', 'value': 'deals'},
                {'label': 'Calls Duration', 'value': 'calls duration'},
            ],
            value='acts',
            clearable=False,  # Убирает крестик для очистки выбора
            style={'width': '300px', 'marginLeft': '10px'}
        )], style={'marginLeft': '0px'}),
    dbc.Row(html.Div([html.H4("by Operator Names", style={'text-align': 'center'})])),
    dbc.Row([
        dbc.Col([
             html.Div([html.Div(id='operator_load_content')], style={'margin': '0'}),           
        ], width={'size': 6}),
        dbc.Col([
            html.Div([html.Div(id='operator_load_category_content')], style={'margin': '0'}),
        ], width={'size': 6}),
        ]),    
        
    
]

#MARKETING
tab4_content = [
    dbc.Row(html.H1('Marketing')),
    html.Br(),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(f"{source['CPA'].mean().round(1)} €", className="card-title"),
            html.P("Cost per Action (CPA)", className="card-text",
            )]
        ),   style={'border': '2px solid #00BFFF',  'box-shadow': '5px 5px 5px rgba(0, 0, 0, 0.2)',   'border-radius': '12px' , 'text-align': 'center'})),
         dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(f"{source['ARPU'].mean().round(1)} €", className="card-title"),
            html.P("Average Revenue Per User (ARPU)", className="card-text",
            ),]
         ),   style={'border': '2px solid #4169E1',  'box-shadow': '5px 5px 5px rgba(0, 0, 0, 0.2)',   'border-radius': '12px' , 'text-align': 'center'})),
        
         dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(f"{source['CRR'].mean().round(1)} %", className="card-title"),
            html.P("Cost Revenue Ratio (CRR)", className="card-text",
            ),]
         ),   style={'border': '2px solid #00BFFF',  'box-shadow': '5px 5px 5px rgba(0, 0, 0, 0.2)',   'border-radius': '12px' , 'text-align': 'center'})),
        
         dbc.Col(dbc.Card(dbc.CardBody([
            html.H3(f"{source['MER'].mean().round(1)} €", className="card-title"),
            html.P("Marketing Efficiency Ratio (MER)", className="card-text",
            ),]
         ),   style={'border': '2px solid #4169E1',  'box-shadow': '5px 5px 5px rgba(0, 0, 0, 0.2)',   'border-radius': '12px' , 'text-align': 'center'})),
        
    ],
    className="mb-4"),
    html.Br(),  
    html.Br(),  
    dbc.Row([html.H5('Top Campaigns by Metrics', style={'text-align': 'center', 'margin-bottom': '24px'}),
            dash_table.DataTable(
            data=campaign.round(1).to_dict('records'),  # Все данные
            columns=[{'id': c, 'name': c, 'type': 'text'} for c in campaign.columns],          
            style_cell={'textAlign': 'center', 'padding': '5px', 'border': '1px solid blue' },                
            style_header={
                'backgroundColor': '#4169E1',  
                'color':'white',
                'fontWeight': 'bold'
            },
             style_data={
            'color': 'black',
            'backgroundColor': '#f4fbf8'
            },
            style_cell_conditional=[{'if': {'column_id': 'Campaign'}, 'width': '20%'}],
            page_size=10,  
            sort_action='native'
            
          
        )]),    
    html.Br(),  
    #dbc.Row(dcc.Graph(figure=campaign_table(), style={'margin': '0px'})),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='metric-dropdown',
                options=[
                    {'label': 'CTR, Convers UA, Convers C1', 'value': 'group1'},
                    {'label': 'CPC, CAC, CPA', 'value': 'group2'},
                    {'label': 'ARPU, CRR, MER', 'value': 'group3'}
                ],
                value='group1',
                clearable=False,
                style={'width': '300px', 'marginLeft': '10px'}
            ),
        ], width={'size': 2}),
        dbc.Col([
             html.H5(id='title_sourse', style={'text-align': 'center'})
        ], width={'size': 8}),
        
        ]),
    dbc.Row(dcc.Graph(id='metric-graph',  style={'margin': '0px'})),
    html.Br(), 
    html.Br(), 
    dbc.Row([
        dbc.Col([
            dcc.Graph(figure=spend_by_month())
        ], width={'size': 8}),
        dbc.Col([
            dcc.Graph(figure=pie_source())
        ], width={'size': 4}),
        
        ]),


    ]
 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
colors = px.colors.diverging.Picnic

app.layout = html.Div([
    dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="About School", tab_style={'marginLeft': 'auto', 'font-size': '14px'}),
        dbc.Tab(tab2_content, label="Deals",  tab_style={'font-size': '14px'}),
        dbc.Tab(tab3_content, label="Operators",  tab_style={'font-size': '14px'}),
        dbc.Tab(tab4_content, label="Marketing",  tab_style={'font-size': '14px'}),
        ], style={'margin-bottom': '12px'}  ),
    

], style={'margin-left': '60px', 'margin-right': '60px','margin-top': '0px'})

@app.callback(
    [Output('metric-graph', 'figure'),
     Output('title_sourse', 'children')],
    [Input('metric-dropdown', 'value')]
)
def update_table(selected_group):
    if selected_group == 'group1':
        value_vars = ['CTR', 'Convers UA', 'Convers C1']
        title_sourse = "Sources: CTR, Convers UA, Convers C1"
    elif selected_group == 'group2':
        value_vars = ['CPC', 'CAC', 'CPA']
        title_sourse = "Sources: CPC, CAC, CPA"
    else:
        value_vars = ['ARPU', 'CRR', 'MER']
        title_sourse = "Sources: ARPU, CRR, MER"

    # Функция для построения графика
    figure = sourses_graph(value_vars)

    # Возвращаем график и заголовок
    return figure, title_sourse
    
@app.callback(
    [Output('table-name', 'children'),
    Output('count_persons', 'children'),
    Output('avg_act_time_content', 'children'),
    Output('title_avg', 'children'),
    Output('category_content', 'children'),
    Output('act_category_time_content', 'children'),
    Output('operator_load_content', 'children'),
    Output('operator_load_category_content', 'children'),],
    [Input('table-dropdown', 'value')]
)

def update_table(selected_table):
    global acts, cnts, calls, deals
    if selected_table == 'acts':
        table_name = 'All Activities'
        count_persons = acts['Owner Name'].nunique()
        df = acts
        title_avg = 'Average Load per Operator'
    elif selected_table == 'cnts':
        table_name = 'Contacts'
        count_persons = cnts['Owner Name'].nunique()
        df = cnts
        title_avg = 'Average Load per Operator'
    elif selected_table == 'calls':
        table_name = 'Calls'
        count_persons = calls['Owner Name'].nunique()
        df = calls
        title_avg = 'Average Load per Operator'
    elif selected_table == 'deals':
        table_name = 'Deals'
        count_persons = deals['Owner Name'].nunique()
        df = deals
        title_avg = 'Average Load per Operator'
    elif selected_table == 'calls duration':
        table_name = 'Calls Duration'
        count_persons = calls['Owner Name'].nunique()
        df = calls
        title_avg = 'Average Call Duration per Operator'
    avg_act_time_content = dcc.Graph(figure=avg_act_time(table_name), style={'height': '400px'})
    category_content = dcc.Graph(figure=create_category_act(df, table_name), style={'height': '240px', 'color_continuous_scale':'px.colors.sequential.Blues'})
    act_category_time_content = dcc.Graph(figure=act_category_time(df, table_name), style={'height': '360px'})
    operator_load_content = dcc.Graph(figure=operator_load(table_name), style={'height': '820px'})
    operator_load_category_content = dcc.Graph(figure=operator_load_category(df, table_name), style={'height': '820px'})

    return table_name, str(count_persons), avg_act_time_content, title_avg, category_content, act_category_time_content, operator_load_content, operator_load_category_content



if __name__ == "__main__":
    if os.environ.get('ENV') == 'production':
		app.run_server(host='0.0.0.0', port=10000, debug=False)  # Для облака
	else:
	    app.run_server(host='127.0.0.1', port=10000, debug=True)  # Для локальной разработки
  