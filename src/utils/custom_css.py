"""
Custom CSS styles for enhanced Streamlit UI
"""

def get_custom_css():
    """Return custom CSS for enhanced UI styling"""
    return """
    <style>
        /* Main container styling */
        .main-container {
            padding: 2rem;
        }
        
        /* Metric containers */
        .metric-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .metric-container h3 {
            margin: 0;
            font-size: 1.2em;
            font-weight: 600;
        }
        
        .metric-container p {
            margin: 5px 0 0 0;
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        /* Trend indicators */
        .bullish {
            color: #00ff00;
            font-weight: bold;
            text-shadow: 0 0 5px rgba(0, 255, 0, 0.3);
        }
        
        .bearish {
            color: #ff0000;
            font-weight: bold;
            text-shadow: 0 0 5px rgba(255, 0, 0, 0.3);
        }
        
        .neutral {
            color: #ffa500;
            font-weight: bold;
            text-shadow: 0 0 5px rgba(255, 165, 0, 0.3);
        }
        
        /* Scan result containers */
        .scan-result {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .scan-result:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        }
        
        /* AI Score based styling */
        .very-strong {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-color: #28a745;
            border-width: 3px;
        }
        
        .strong {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            border-color: #17a2b8;
            border-width: 2px;
        }
        
        .moderate {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border-color: #ffc107;
            border-width: 2px;
        }
        
        .weak {
            background: linear-gradient(135deg, #f8d7da 0%, #f1b0b7 100%);
            border-color: #dc3545;
            border-width: 2px;
        }
        
        .very-weak {
            background: linear-gradient(135deg, #f8d7da 0%, #e2a8a8 100%);
            border-color: #721c24;
            border-width: 3px;
        }
        
        /* Stock symbol styling */
        .stock-symbol {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        }
        
        /* AI Score display */
        .ai-score {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            border-radius: 50%;
            display: inline-block;
            min-width: 80px;
            min-height: 80px;
            line-height: 60px;
            margin: 10px;
        }
        
        .ai-score-very-strong {
            background: radial-gradient(circle, #28a745, #1e7e34);
            color: white;
            box-shadow: 0 0 20px rgba(40, 167, 69, 0.4);
        }
        
        .ai-score-strong {
            background: radial-gradient(circle, #17a2b8, #138496);
            color: white;
            box-shadow: 0 0 20px rgba(23, 162, 184, 0.4);
        }
        
        .ai-score-moderate {
            background: radial-gradient(circle, #ffc107, #e0a800);
            color: white;
            box-shadow: 0 0 20px rgba(255, 193, 7, 0.4);
        }
        
        .ai-score-weak {
            background: radial-gradient(circle, #dc3545, #c82333);
            color: white;
            box-shadow: 0 0 20px rgba(220, 53, 69, 0.4);
        }
        
        /* Price change indicators */
        .price-up {
            color: #28a745;
            font-weight: bold;
        }
        
        .price-up::before {
            content: "▲ ";
        }
        
        .price-down {
            color: #dc3545;
            font-weight: bold;
        }
        
        .price-down::before {
            content: "▼ ";
        }
        
        .price-neutral {
            color: #6c757d;
            font-weight: bold;
        }
        
        .price-neutral::before {
            content: "● ";
        }
        
        /* Progress bars */
        .progress-container {
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-bar {
            height: 20px;
            background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dc3545 100%);
            border-radius: 10px;
            transition: width 0.5s ease;
        }
        
        /* Signal strength indicators */
        .signal-very-strong {
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: bold;
            display: inline-block;
            margin: 2px;
            box-shadow: 0 2px 5px rgba(40, 167, 69, 0.3);
        }
        
        .signal-strong {
            background: linear-gradient(45deg, #17a2b8, #20c997);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: bold;
            display: inline-block;
            margin: 2px;
            box-shadow: 0 2px 5px rgba(23, 162, 184, 0.3);
        }
        
        .signal-moderate {
            background: linear-gradient(45deg, #ffc107, #fd7e14);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: bold;
            display: inline-block;
            margin: 2px;
            box-shadow: 0 2px 5px rgba(255, 193, 7, 0.3);
        }
        
        .signal-weak {
            background: linear-gradient(45deg, #dc3545, #e83e8c);
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: bold;
            display: inline-block;
            margin: 2px;
            box-shadow: 0 2px 5px rgba(220, 53, 69, 0.3);
        }
        
        /* Loading animations */
        .loading-spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Card styling */
        .info-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #007bff;
        }
        
        .warning-card {
            background: #fff3cd;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #ffc107;
        }
        
        .success-card {
            background: #d4edda;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #28a745;
        }
        
        .error-card {
            background: #f8d7da;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #dc3545;
        }
        
        /* Table styling */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #5a67d8 0%, #667eea 100%);
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .metric-container {
                margin: 5px 0;
                padding: 10px;
            }
            
            .stock-symbol {
                font-size: 1.2em;
            }
            
            .ai-score {
                font-size: 1.5em;
                min-width: 60px;
                min-height: 60px;
                line-height: 40px;
            }
        }
        
        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
    </style>
    """

def get_signal_emoji(signal_type: str) -> str:
    """Get emoji for signal types"""
    emoji_map = {
        'STRONG BUY': '🚀',
        'BUY': '🔥',
        'BUY DIP': '💎',
        'STRONG SELL': '💥',
        'SELL': '📉',
        'SELL RALLY': '⚡',
        'WAIT': '⏳',
        'AVOID': '❌',
        'INSUFFICIENT_DATA': '❓',
        'ERROR': '⚠️'
    }
    return emoji_map.get(signal_type, '❓')

def get_trend_color(trend: str) -> str:
    """Get color for trend indicators"""
    color_map = {
        'Bullish': '#28a745',
        'Bearish': '#dc3545',
        'Neutral': '#ffc107',
        'bullish': '#28a745',
        'bearish': '#dc3545',
        'neutral': '#ffc107'
    }
    return color_map.get(trend, '#6c757d')

def format_ai_score_class(score: float) -> str:
    """Get CSS class based on AI score"""
    if score >= 75:
        return 'ai-score-very-strong'
    elif score >= 60:
        return 'ai-score-strong'
    elif score >= 45:
        return 'ai-score-moderate'
    else:
        return 'ai-score-weak'

def format_signal_class(signal: str) -> str:
    """Get CSS class based on signal strength"""
    signal_lower = signal.lower()
    if 'very strong' in signal_lower or 'strong buy' in signal_lower or 'strong sell' in signal_lower:
        return 'signal-very-strong'
    elif 'strong' in signal_lower:
        return 'signal-strong'
    elif 'moderate' in signal_lower:
        return 'signal-moderate'
    else:
        return 'signal-weak'