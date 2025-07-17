# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
import yfinance as yf
import vnstock as vs
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import datetime, timedelta
import requests

# --- CẤU HÌNH TRANG & CSS ---
st.set_page_config(layout="wide", page_title="Dashboard Phân Tích Cổ Phiếu Pro")

st.markdown("""
<style>
    /* Font chữ chính */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Thanh sidebar */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    /* Tiêu đề chính */
    h1 {
        font-size: 2.5em;
        font-weight: 700;
        letter-spacing: -1px;
        color: #1e3c72; /* Dark blue */
    }
    /* Tiêu đề tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #f0f2f6;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 16px;
        font-weight: 600;
        font-size: 1.1em;
        color: #4f4f4f;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e8eaf6; /* Light blue */
        color: #1e3c72; /* Dark blue */
        border-bottom: 2px solid #3f51b5; /* Indigo */
    }
    /* Card số liệu */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-card h4 {
        font-size: 1em;
        font-weight: 400;
        color: #616161; /* Medium grey */
        margin-bottom: 8px;
    }
    .metric-card p {
        font-size: 1.5em;
        font-weight: 700;
        color: #212121; /* Dark grey */
        margin: 0;
    }
    /* Tín hiệu mua/bán */
    .signal {
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 15px;
        color: white;
        display: inline-block;
    }
    .strong-buy { background-color: #006400; } /* DarkGreen */
    .buy { background-color: #4CAF50; } /* Green */
    .neutral { background-color: #FFC107; } /* Amber */
    .sell { background-color: #F44336; } /* Red */
    .strong-sell { background-color: #B71C1C; } /* DarkRed */

</style>
""", unsafe_allow_html=True)


# --- CÁC HÀM LẤY DỮ LIỆU (CÓ CACHE) ---

@st.cache_data(ttl=300) # Cache dữ liệu trong 5 phút
def fetch_stock_data(symbol, start_date, end_date):
    """
    Lấy dữ liệu cổ phiếu, thông tin công ty, và tin tức từ yfinance hoặc vnstock.
    """
    is_vn_stock = symbol.upper().endswith('.VN')
    if is_vn_stock:
        symbol_root = symbol.upper().replace('.VN', '')
        try:
            df = vs.stock_historical_data(symbol_root, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if df.empty:
                return None, f"Không tìm thấy dữ liệu cho mã VN: {symbol_root}"
            # Chuẩn hóa cột: vnstock có thể dùng 'time' hoặc 'TradingDate'
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'Date'})
            elif 'TradingDate' in df.columns:
                df = df.rename(columns={'TradingDate': 'Date'})
            else:
                 return None, "Không tìm thấy cột ngày tháng trong dữ liệu vnstock."

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.columns = [col.capitalize() for col in df.columns]

            # Lấy thông tin công ty từ vnstock (ít chi tiết hơn yfinance)
            overview = vs.company_overview(symbol_root)
            info = {
                'longName': overview['companyName'].iloc[0] if not overview.empty else symbol_root,
                'longBusinessSummary': overview['companyProfile'].iloc[0] if not overview.empty else "Không có mô tả.",
                'marketCap': overview['marketCap'].iloc[0] * 1_000_000_000 if not overview.empty else None,
                'trailingPE': overview['priceToEarningsRatio'].iloc[0] if not overview.empty else None,
                'beta': overview['beta'].iloc[0] if not overview.empty else None,
                'industry': overview['industry'].iloc[0] if not overview.empty else "N/A"
            }
            news = [] # vnstock không có API tin tức, sẽ dùng Finnhub
        except Exception as e:
            return None, f"Lỗi khi tải dữ liệu từ vnstock: {e}"
    else: # Cổ phiếu quốc tế
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info or 'longName' not in info:
                return None, f"Mã cổ phiếu quốc tế không hợp lệ: {symbol}"
            df = ticker.history(start=start_date, end=end_date)
            if df.empty:
                return None, f"Không tìm thấy dữ liệu lịch sử cho {symbol} trong khoảng thời gian đã chọn."
            news = ticker.news
        except Exception as e:
            return None, f"Lỗi khi tải dữ liệu từ yfinance: {e}"

    # Lấy tin tức từ Finnhub nếu là mã VN hoặc yfinance không có tin tức
    if is_vn_stock or not news:
        try:
            API_KEY = 'd1rsj2pr01qm5ddsjamgd1rsj2pr01qm5ddsjan0' # Khóa API Finnhub
            finnhub_symbol = symbol.replace('.VN', '')
            end_date_str = datetime.now().strftime('%Y-%m-%d')
            start_date_str = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            url = f'https://finnhub.io/api/v1/company-news?symbol={finnhub_symbol}&from={start_date_str}&to={end_date_str}&token={API_KEY}'
            r = requests.get(url)
            r.raise_for_status()
            news = r.json()
        except Exception as e:
            st.sidebar.warning(f"Không thể tải tin tức từ Finnhub: {e}")
            news = []

    return {'df': df, 'info': info, 'news': news}, None

@st.cache_data
def get_financials(symbol):
    is_vn_stock = symbol.upper().endswith('.VN')
    symbol_root = symbol.upper().replace('.VN', '')
    try:
        if is_vn_stock:
            income = vs.financial_flow(symbol_root, 'incomestatement', 'quarterly')
            balance = vs.financial_flow(symbol_root, 'balancesheet', 'quarterly')
            cashflow = vs.financial_flow(symbol_root, 'cashflow', 'quarterly')
        else:
            ticker = yf.Ticker(symbol)
            income = ticker.income_stmt
            balance = ticker.balance_sheet
            cashflow = ticker.cashflow
        return {'income': income, 'balance': balance, 'cashflow': cashflow}
    except Exception:
        return None

@st.cache_data
def get_technical_summary(df):
    """Tính toán các chỉ báo và đưa ra khuyến nghị tổng hợp."""
    if df.empty or len(df) < 20:
        return None, "Dữ liệu không đủ để phân tích kỹ thuật."

    summary = {'MA': 0, 'OSC': 0}
    buy_count = 0
    sell_count = 0
    neutral_count = 0

    # Moving Averages Signals
    for period in [10, 20, 50]:
        sma = ta.sma(df['Close'], length=period)
        if sma is not None and not sma.empty:
            if df['Close'].iloc[-1] > sma.iloc[-1]:
                buy_count += 1
            else:
                sell_count += 1
    summary['MA'] = 'Buy' if buy_count > sell_count else 'Sell'

    # Oscillators Signals
    buy_count, sell_count, neutral_count = 0, 0, 0
    rsi = ta.rsi(df['Close'], length=14)
    if rsi is not None and not rsi.empty:
        if rsi.iloc[-1] < 30: buy_count += 1
        elif rsi.iloc[-1] > 70: sell_count += 1
        else: neutral_count += 1

    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    if stoch is not None and not stoch.empty:
        if stoch['STOCHk_14_3_3'].iloc[-1] < 20 and stoch['STOCHd_14_3_3'].iloc[-1] < 20: buy_count +=1
        elif stoch['STOCHk_14_3_3'].iloc[-1] > 80 and stoch['STOCHd_14_3_3'].iloc[-1] > 80: sell_count += 1
        else: neutral_count += 1

    macd = ta.macd(df['Close'])
    if macd is not None and not macd.empty:
        if macd['MACD_12_26_9'].iloc[-1] > macd['MACDs_12_26_9'].iloc[-1]: buy_count += 1
        else: sell_count += 1

    summary['OSC'] = 'Buy' if buy_count > sell_count else ('Sell' if sell_count > buy_count else 'Neutral')

    # Overall Summary
    total_score = (summary['MA'] == 'Buy') + (summary['OSC'] == 'Buy') - (summary['MA'] == 'Sell') - (summary['OSC'] == 'Sell')
    if total_score == 2: return "Mua Mạnh", summary
    if total_score == 1: return "Mua", summary
    if total_score == -1: return "Bán", summary
    if total_score == -2: return "Bán Mạnh", summary
    return "Trung Lập", summary

@st.cache_data
def get_prediction(df):
    """Dự báo giá sử dụng Facebook Prophet."""
    if len(df) < 30:
        return None, "Cần ít nhất 30 ngày dữ liệu để dự báo."
    df_train = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True,
                    changepoint_prior_scale=0.05, seasonality_prior_scale=10.0)
    model.fit(df_train)
    
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    return forecast, model

# --- GIAO DIỆN NGƯỜI DÙNG ---

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    symbol_input = st.text_input("Nhập mã cổ phiếu (VD: NVDA, FPT.VN)", "FPT.VN").upper()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Từ ngày", datetime.now() - timedelta(days=365*2))
    with col2:
        end_date = st.date_input("Đến ngày", datetime.now())

    if st.button("🚀 Phân Tích", use_container_width=True):
        if not symbol_input:
            st.error("Vui lòng nhập mã cổ phiếu.")
        elif start_date >= end_date:
            st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        else:
            with st.spinner(f"Đang tải và phân tích {symbol_input}..."):
                data, error = fetch_stock_data(symbol_input, start_date, end_date)
                if error:
                    st.error(error)
                    st.session_state.stock_data = None
                else:
                    st.session_state.stock_data = data
                    st.session_state.symbol = symbol_input
    
    if 'stock_data' in st.session_state and st.session_state.stock_data:
        st.success(f"Đã tải xong dữ liệu cho {st.session_state.symbol}")


# --- Main Content ---
st.title(f"📊 Dashboard Phân Tích Cổ Phiếu Pro")

if 'stock_data' not in st.session_state or not st.session_state.stock_data:
    st.info("Chào mừng bạn đến với Dashboard Phân Tích Cổ Phiếu. Vui lòng nhập mã và nhấn 'Phân Tích' để bắt đầu.")
else:
    # Lấy dữ liệu từ session_state
    data = st.session_state.stock_data
    symbol = st.session_state.symbol
    df = data['df']
    info = data['info']
    
    # --- Header ---
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100
    change_color = "green" if change >= 0 else "red"
    
    st.markdown(f"## {info.get('longName', symbol)} ({symbol})")
    st.markdown(f"<h2 style='color:{change_color};'>{current_price:,.2f} "
                f"<span style='font-size:0.7em;'>({change:+.2f} / {change_pct:+.2f}%)</span></h2>", 
                unsafe_allow_html=True)
    
    # --- Tabs ---
    tab_list = ["Tổng quan", "Phân tích Kỹ thuật", "Dữ liệu Tài chính", "Dự báo Prophet", "So sánh Ngành", "Tin tức"]
    overview_tab, tech_tab, fun_tab, forecast_tab, peer_tab, news_tab = st.tabs(tab_list)

    with overview_tab:
        st.subheader("📈 Biểu đồ giá và Khối lượng")
        
        # Tạo biểu đồ với 2 trục y
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=('Giá', 'Khối lượng'), 
                            row_width=[0.2, 0.7])

        # Biểu đồ nến
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'], name='Giá'), 
                      row=1, col=1)

        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20)
        fig.add_trace(go.Scatter(x=df.index, y=bbands['BBU_20_2.0'], name='Upper Band', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bbands['BBM_20_2.0'], name='Middle Band', line=dict(color='orange', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bbands['BBL_20_2.0'], name='Lower Band', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)

        # Biểu đồ khối lượng
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Khối lượng', marker_color='rgba(79, 79, 79, 0.5)'), row=2, col=1)

        fig.update_layout(showlegend=False, height=600,
                          xaxis_rangeslider_visible=False,
                          yaxis_title='Giá (USD/VND)',
                          yaxis2_title='Khối lượng')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🔑 Các chỉ số chính")
        market_cap = info.get('marketCap', 0)
        market_cap_str = f"{market_cap/1e12:,.2f} Tỷ tỷ" if market_cap > 1e12 else f"{market_cap/1e9:,.2f} Tỷ"
        
        metrics = {
            "Vốn hóa thị trường": market_cap_str,
            "P/E Ratio": f"{info.get('trailingPE'):,.2f}" if info.get('trailingPE') else "N/A",
            "Beta": f"{info.get('beta'):,.2f}" if info.get('beta') else "N/A",
            "EPS": f"{info.get('trailingEps'):,.2f}" if info.get('trailingEps') else "N/A",
            "52-Tuần Cao": f"{info.get('fiftyTwoWeekHigh'):,.2f}" if info.get('fiftyTwoWeekHigh') else "N/A",
            "52-Tuần Thấp": f"{info.get('fiftyTwoWeekLow'):,.2f}" if info.get('fiftyTwoWeekLow') else "N/A"
        }
        
        cols = st.columns(3)
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i%3]:
                st.markdown(f"<div class='metric-card'><h4>{label}</h4><p>{value}</p></div>", unsafe_allow_html=True)
                st.write("") # Add space

        st.subheader("🏢 Hồ sơ công ty")
        st.write(f"**Ngành:** {info.get('industry', 'N/A')}")
        st.write(f"**Website:** {info.get('website', 'N/A')}")
        with st.expander("Xem mô tả chi tiết"):
            st.write(info.get('longBusinessSummary', 'Không có mô tả chi tiết.'))

    with tech_tab:
        st.subheader("📊 Tóm tắt Phân tích Kỹ thuật")
        summary, details = get_technical_summary(df)

        # Đồng hồ đo
        gauge_value = {'Bán Mạnh': 10, 'Bán': 30, 'Trung Lập': 50, 'Mua': 70, 'Mua Mạnh': 90}.get(summary, 50)
        color_map = {'Bán Mạnh': '#B71C1C', 'Bán': '#F44336', 'Trung Lập': '#FFC107', 'Mua': '#4CAF50', 'Mua Mạnh': '#006400'}
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = gauge_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Khuyến nghị: {summary}", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color_map[summary]},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#B71C1C'},
                    {'range': [20, 40], 'color': '#F44336'},
                    {'range': [40, 60], 'color': '#FFC107'},
                    {'range': [60, 80], 'color': '#4CAF50'},
                    {'range': [80, 100], 'color': '#006400'}],
            }))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.subheader("📈 Chi tiết các chỉ báo")
        # Chia cột để hiển thị RSI, MACD
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### RSI (Relative Strength Index)")
            rsi = ta.rsi(df['Close'])
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Quá mua")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Quá bán")
            fig_rsi.update_layout(height=300, yaxis_range=[0,100], margin=dict(t=20, b=20))
            st.plotly_chart(fig_rsi, use_container_width=True)

        with col2:
            st.markdown("#### MACD")
            macd = ta.macd(df['Close'])
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(x=df.index, y=macd['MACD_12_26_9'], name='MACD', line=dict(color='blue')))
            fig_macd.add_trace(go.Scatter(x=df.index, y=macd['MACDs_12_26_9'], name='Signal', line=dict(color='orange')))
            fig_macd.add_trace(go.Bar(x=df.index, y=macd['MACDh_12_26_9'], name='Histogram', marker_color=np.where(macd['MACDh_12_26_9'] < 0, 'red', 'green')))
            fig_macd.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_macd, use_container_width=True)

    with fun_tab:
        st.subheader("📑 Báo cáo Tài chính")
        financials = get_financials(symbol)
        if financials:
            inc_tab, bal_tab, cf_tab = st.tabs(["Báo cáo KQKD", "Bảng Cân đối Kế toán", "Lưu chuyển Tiền tệ"])
            with inc_tab:
                st.dataframe(financials['income'])
            with bal_tab:
                st.dataframe(financials['balance'])
            with cf_tab:
                st.dataframe(financials['cashflow'])
        else:
            st.warning("Không thể tải dữ liệu tài chính cho mã này.")
            
    with forecast_tab:
        st.subheader("🔮 Dự báo Giá bằng Prophet")
        with st.spinner("Đang huấn luyện mô hình và dự báo..."):
            forecast, model = get_prediction(df)

        if forecast is not None:
            fig = plot_plotly(model, forecast)
            fig.update_layout(title_text='Dự báo giá 90 ngày tới',
                              xaxis_title='Ngày', yaxis_title='Giá')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Chi tiết dự báo:")
            forecast_summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(90)
            forecast_summary = forecast_summary.rename(columns={
                'ds':'Ngày', 'yhat':'Giá dự báo', 'yhat_lower':'Dự báo thấp', 'yhat_upper':'Dự báo cao'
            })
            st.dataframe(forecast_summary.set_index('Ngày'))
            st.warning("⚠️ **Lưu ý:** Đây là dự báo từ mô hình toán học và chỉ mang tính tham khảo, không phải lời khuyên đầu tư.")
        else:
            st.error("Không đủ dữ liệu để tạo dự báo.")
            
    with peer_tab:
        st.subheader("🏢 So sánh hiệu suất với các đối thủ")
        peers_input = st.text_input("Nhập mã các cổ phiếu khác để so sánh (cách nhau bởi dấu phẩy)", "NVDA, AMD, INTC")
        
        if peers_input:
            peers = [p.strip().upper() for p in peers_input.split(',')]
            all_symbols = [symbol] + peers
            
            with st.spinner("Đang tải dữ liệu so sánh..."):
                all_data = {}
                for s in all_symbols:
                    peer_data, _ = fetch_stock_data(s, start_date, end_date)
                    if peer_data:
                        all_data[s] = peer_data['df']['Close']
            
            if all_data:
                normalized_df = pd.DataFrame({s: (df / df.iloc[0] - 1) * 100 for s, df in all_data.items()})
                
                fig = go.Figure()
                for col in normalized_df.columns:
                    fig.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df[col], name=col))
                
                fig.update_layout(title='So sánh Hiệu suất Giá (Chuẩn hóa %)',
                                  xaxis_title='Ngày', yaxis_title='Thay đổi (%)',
                                  height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Không tải được dữ liệu cho các mã so sánh.")

    with news_tab:
        st.subheader("📰 Tin tức mới nhất")
        news = data['news']
        if news:
            for item in news[:10]: # Hiển thị 10 tin mới nhất
                # Sửa lỗi key có thể không tồn tại trong dict trả về từ API
                title = item.get('headline', 'Không có tiêu đề')
                link = item.get('url', '#')
                source = item.get('source', 'Không rõ nguồn')
                summary = item.get('summary', 'Không có tóm tắt.')
                
                # Chuyển đổi datetime từ timestamp nếu có
                dt_object = "N/A"
                if 'datetime' in item:
                    try:
                        dt_object = datetime.fromtimestamp(item['datetime']).strftime('%d-%m-%Y')
                    except:
                        pass # Bỏ qua nếu timestamp không hợp lệ
                
                st.markdown(f"**<a href='{link}' target='_blank'>{title}</a>**", unsafe_allow_html=True)
                st.caption(f"Nguồn: {source} - Ngày: {dt_object}")
                st.write(summary)
                st.markdown("---")
        else:
            st.info(f"Không tìm thấy tin tức gần đây cho {symbol}.")
