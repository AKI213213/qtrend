# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 17:42:48 2025

@author: redmiG2021
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import base64
from datetime import datetime
import plotly.graph_objects as go
from io import BytesIO
import zipfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from matplotlib import font_manager
import os
import sys

# 设置matplotlib使用非交互式后端
matplotlib.use('Agg')

# 尝试添加中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    try:
        # 尝试查找系统中的中文字体
        font_dirs = [
            '/usr/share/fonts',  # Linux通用
            '/usr/local/share/fonts',  # Linux本地
            'C:/Windows/Fonts',  # Windows
            'C:/Windows/Fonts',  # Windows备用
            '/System/Library/Fonts',  # macOS
            '/Library/Fonts',  # macOS
        ]
        
        # 常见中文字体名称
        chinese_fonts = [
            'simhei.ttf',  # 黑体
            'simsun.ttc',  # 宋体
            'msyh.ttc',  # 微软雅黑
            'msyh.ttf',  # 微软雅黑
            'STKAITI.TTF',  # 楷体
            'STSONG.TTF',  # 宋体
            'DroidSansFallback.ttf',  # Android回退字体
            'DejaVuSans.ttf',  # 通用字体
        ]
        
        # 查找可用的中文字体
        found_font = None
        
        for font_dir in font_dirs:
            if os.path.exists(font_dir):
                for font_file in chinese_fonts:
                    font_path = os.path.join(font_dir, font_file)
                    if os.path.exists(font_path):
                        found_font = font_path
                        break
                if found_font:
                    break
        
        if found_font:
            # 添加字体到matplotlib
            font_manager.fontManager.addfont(found_font)
            font_name = font_manager.FontProperties(fname=found_font).get_name()
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            return True
        else:
            # 如果找不到中文字体，尝试使用内置字体
            try:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                return True
            except:
                return False
    except Exception as e:
        print(f"设置中文字体时出错: {e}", file=sys.stderr)
        return False

# 初始化中文字体
setup_chinese_font()

# ------------------------------
# 页面配置
# ------------------------------
st.set_page_config(
    page_title="学生成绩查询系统", 
    layout="wide",
    page_icon="🎓"
)

# 页面标题
st.title("🎓 学生成绩查询系统")
st.markdown("上传包含大量学生成绩的Excel文件，通过班级和姓名快速查询学生各次考试成绩。")

# ------------------------------
# 初始化会话状态
# ------------------------------
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'class_column_name' not in st.session_state:
    st.session_state.class_column_name = '班别'
if 'name_column_name' not in st.session_state:
    st.session_state.name_column_name = '姓名'
if 'id_column_name' not in st.session_state:
    st.session_state.id_column_name = '学籍号'
if 'subjects' not in st.session_state:
    st.session_state.subjects = []
if 'exams' not in st.session_state:
    st.session_state.exams = []
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'selected_viz_subjects' not in st.session_state:
    st.session_state.selected_viz_subjects = []
if 'grades_df' not in st.session_state:
    st.session_state.grades_df = None
if 'current_student' not in st.session_state:
    st.session_state.current_student = None
if 'chart_updated' not in st.session_state:
    st.session_state.chart_updated = True
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'batch_student_grades' not in st.session_state:
    st.session_state.batch_student_grades = {}
if 'batch_global_subjects' not in st.session_state:
    st.session_state.batch_global_subjects = []
if 'show_batch_charts' not in st.session_state:
    st.session_state.show_batch_charts = False
if 'batch_charts_generated' not in st.session_state:
    st.session_state.batch_charts_generated = False
if 'batch_student_charts' not in st.session_state:
    st.session_state.batch_student_charts = {}
if 'batch_query_executed' not in st.session_state:
    st.session_state.batch_query_executed = False
if 'charts_per_page_value' not in st.session_state:
    st.session_state.charts_per_page_value = 6
if 'single_pdf_created' not in st.session_state:
    st.session_state.single_pdf_created = False
if 'single_pdf_data' not in st.session_state:
    st.session_state.single_pdf_data = None
if 'show_rankings' not in st.session_state:
    st.session_state.show_rankings = False
if 'batch_show_rankings' not in st.session_state:
    st.session_state.batch_show_rankings = False
if 'batch_query_mode' not in st.session_state:
    st.session_state.batch_query_mode = "manual"  # "manual" 或 "class_bulk"
if 'selected_batch_classes' not in st.session_state:
    st.session_state.selected_batch_classes = []

# ------------------------------
# 数据处理函数
# ------------------------------
@st.cache_data(ttl=3600)
def load_data(uploaded_file):
    """加载并缓存Excel数据"""
    try:
        df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"文件读取失败: {str(e)}")
        return None

def detect_column_names(df):
    """自动检测列名"""
    column_names = {}
    
    # 检测可能的班级列名
    class_column_candidates = ['班别', '班级', '班', 'Class', 'class', 'CLS', 'cls']
    for col in df.columns:
        for candidate in class_column_candidates:
            if candidate in str(col):
                column_names['class'] = col
                break
        if 'class' in column_names:
            break
    
    # 如果没找到，尝试找只包含数字的列名
    if 'class' not in column_names:
        for col in df.columns:
            if str(col).strip().isdigit():
                column_names['class'] = col
                break
    
    # 检测姓名列
    name_column_candidates = ['姓名', 'Name', 'name', '学生姓名', '学生名']
    for col in df.columns:
        for candidate in name_column_candidates:
            if candidate in str(col):
                column_names['name'] = col
                break
        if 'name' in column_names:
            break
    
    # 检测学籍号列
    id_column_candidates = ['学籍号', '学号', 'ID', 'id', 'StudentID', 'student_id']
    for col in df.columns:
        for candidate in id_column_candidates:
            if candidate in str(col):
                column_names['id'] = col
                break
        if 'id' in column_names:
            break
    
    return column_names

def extract_subjects_exams(df_columns, info_columns):
    """从列名中智能提取科目和考试场次"""
    # 基础信息列
    base_columns = info_columns
    
    # 提取所有非基础列
    grade_columns = [col for col in df_columns if col not in base_columns]
    
    if not grade_columns:
        return [], [], {}
    
    # 定义已知科目列表
    known_subjects = ['语文', '数学', '外语', '政治', '历史', '地理', 
                     '物理', '化学', '生物', '三总', '三排', '总分', '总排']
    
    # 先尝试精确匹配已知科目
    subjects = set()
    column_mapping = {}  # 存储列名到(科目, 考试场次)的映射
    
    for col in grade_columns:
        matched = False
        for subject in known_subjects:
            if col.startswith(subject):
                exam_part = col[len(subject):]
                if exam_part:
                    subjects.add(subject)
                    column_mapping[col] = (subject, exam_part)
                    matched = True
                    break
        
        # 如果没有匹配到已知科目，尝试用正则表达式匹配
        if not matched:
            match = re.match(r'^([\u4e00-\u9fa5]+)(.*)$', col)
            if match:
                subject = match.group(1)
                exam_part = match.group(2)
                if subject and exam_part:
                    subjects.add(subject)
                    column_mapping[col] = (subject, exam_part)
    
    # 对科目进行排序
    subject_order = ['语文', '数学', '外语', '政治', '历史', '地理', 
                    '物理', '化学', '生物', '三总', '三排', '总分', '总排']
    
    sorted_subjects = []
    for priority in subject_order:
        if priority in subjects:
            sorted_subjects.append(priority)
            subjects.discard(priority)
    
    # 添加剩余的科目
    sorted_subjects.extend(sorted(subjects))
    
    # 提取所有考试场次
    exams = set()
    for subject, exam in column_mapping.values():
        exams.add(exam)
    
    # 对考试场次进行排序
    def exam_sort_key(exam):
        semester_order = {'一': 1, '二': 2, '三': 3, '四': 4}
        exam_type_order = {
            '期中': 1, '五校': 2, '期末': 3, 
            '八校': 4, '八月': 5, '九月': 6, '十月': 7, '十二月': 8
        }
        
        semester_match = re.search(r'([一二三])', exam)
        if semester_match:
            semester = semester_order.get(semester_match.group(1), 99)
        else:
            semester = 99
        
        exam_type = None
        for etype in exam_type_order:
            if etype in exam:
                exam_type = etype
                break
        
        if exam_type:
            exam_type_rank = exam_type_order.get(exam_type, 99)
        else:
            exam_type_rank = 99
        
        return (semester, exam_type_rank, exam)
    
    sorted_exams = sorted(exams, key=exam_sort_key)
    
    return sorted_subjects, sorted_exams, column_mapping

def get_student_grades(df, class_name, student_name, class_col, name_col, subjects, exams, column_mapping):
    """获取指定学生的成绩数据"""
    # 确保班级名称类型一致
    df_class_col = df[class_col].astype(str).str.strip()
    input_class_name = str(class_name).strip()
    
    # 筛选学生数据
    mask = (df_class_col == input_class_name) & (df[name_col] == student_name)
    student_data = df[mask]
    
    if student_data.empty:
        return None
    
    student_row = student_data.iloc[0]
    
    # 构建结果DataFrame
    result_data = []
    
    # 为每个考试场次创建一行
    for exam in exams:
        row = {'考试场次': exam}
        
        # 为每个科目填充成绩
        for subject in subjects:
            # 查找对应的列名
            col_name = None
            for col, (subj, exm) in column_mapping.items():
                if subj == subject and exm == exam:
                    col_name = col
                    break
            
            if col_name and col_name in student_row:
                value = student_row[col_name]
                if pd.isna(value):
                    row[subject] = None
                else:
                    try:
                        row[subject] = float(value)
                    except (ValueError, TypeError):
                        row[subject] = str(value).strip()
            else:
                row[subject] = None
        
        result_data.append(row)
    
    return pd.DataFrame(result_data)

def get_batch_student_grades(df, batch_data, class_col, name_col, subjects, exams, column_mapping):
    """批量获取多个学生的成绩数据"""
    all_results = []
    found_students = []
    not_found_students = []
    student_grades_dict = {}
    
    for class_name, student_name in batch_data:
        # 确保班级名称类型一致
        df_class_col = df[class_col].astype(str).str.strip()
        input_class_name = str(class_name).strip()
        
        # 筛选学生数据
        mask = (df_class_col == input_class_name) & (df[name_col] == student_name)
        student_data = df[mask]
        
        if not student_data.empty:
            student_row = student_data.iloc[0]
            found_students.append((class_name, student_name))
            
            # 为每个考试场次创建一行
            for exam in exams:
                row = {'班级': class_name, '姓名': student_name, '考试场次': exam}
                
                # 为每个科目填充成绩
                for subject in subjects:
                    col_name = None
                    for col, (subj, exm) in column_mapping.items():
                        if subj == subject and exm == exam:
                            col_name = col
                            break
                    
                    if col_name and col_name in student_row:
                        value = student_row[col_name]
                        if pd.isna(value):
                            row[subject] = None
                        else:
                            try:
                                row[subject] = float(value)
                            except (ValueError, TypeError):
                                row[subject] = str(value).strip()
                    else:
                        row[subject] = None
                
                all_results.append(row)
            
            # 保存每个学生的单独成绩表
            student_result_data = []
            for exam in exams:
                student_row_data = {'考试场次': exam}
                for subject in subjects:
                    col_name = None
                    for col, (subj, exm) in column_mapping.items():
                        if subj == subject and exm == exam:
                            col_name = col
                            break
                    
                    if col_name and col_name in student_row:
                        value = student_row[col_name]
                        if pd.isna(value):
                            student_row_data[subject] = None
                        else:
                            try:
                                student_row_data[subject] = float(value)
                            except (ValueError, TypeError):
                                student_row_data[subject] = str(value).strip()
                    else:
                        student_row_data[subject] = None
                student_result_data.append(student_row_data)
            
            student_grades_dict[f"{class_name}_{student_name}"] = pd.DataFrame(student_result_data)
        else:
            not_found_students.append((class_name, student_name))
    
    if all_results:
        return pd.DataFrame(all_results), found_students, not_found_students, student_grades_dict
    else:
        return None, [], not_found_students, {}

def get_class_all_students(df, class_names, class_col, name_col):
    """获取指定班级的所有学生名单"""
    batch_data = []
    
    for class_name in class_names:
        # 确保班级名称类型一致
        df_class_col = df[class_col].astype(str).str.strip()
        input_class_name = str(class_name).strip()
        
        # 筛选指定班级的学生
        mask = (df_class_col == input_class_name)
        class_students = df[mask][name_col].dropna().unique()
        
        for student_name in class_students:
            batch_data.append((class_name, student_name))
    
    return batch_data

def create_grade_trend_chart(grades_df, subjects_to_plot, student_name="", class_name=""):
    """创建成绩趋势图表（Plotly版本），在数据点上显示数值"""
    if grades_df.empty or not subjects_to_plot:
        return None
    
    # 准备数据
    chart_data = grades_df[['考试场次'] + subjects_to_plot].copy()
    
    # 转换为数值类型
    for subject in subjects_to_plot:
        if subject in chart_data.columns:
            chart_data[subject] = pd.to_numeric(chart_data[subject], errors='coerce')
    
    # 创建图表
    fig = go.Figure()
    
    # 为每个科目添加一条线
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
              '#3182bd', '#e6550d', '#31a354', '#756bb1', '#636363']
    
    for idx, subject in enumerate(subjects_to_plot):
        if subject in chart_data.columns:
            color_idx = idx % len(colors)
            y_values = chart_data[subject].values
            
            fig.add_trace(go.Scatter(
                x=chart_data['考试场次'],
                y=y_values,
                mode='lines+markers+text',
                name=subject,
                line=dict(color=colors[color_idx], width=3),
                marker=dict(size=8, color=colors[color_idx]),
                text=[f'{y:.1f}' if not np.isnan(y) else '' for y in y_values],
                textposition='top center',
                textfont=dict(size=10, color=colors[color_idx])
            ))
    
    # 更新图表布局
    title = f"{student_name} 成绩趋势图" if student_name else "成绩趋势图"
    if class_name:
        title = f"{class_name} - {title}"
    
    fig.update_layout(
        title=title,
        xaxis_title='考试场次',
        yaxis_title='成绩',
        height=500 + (len(subjects_to_plot) - 3) * 30,
        hovermode='x unified',
        legend=dict(
            orientation="v" if len(subjects_to_plot) > 6 else "h",
            yanchor="middle" if len(subjects_to_plot) > 6 else "bottom",
            y=1 if len(subjects_to_plot) > 6 else 1.02,
            xanchor="left" if len(subjects_to_plot) > 6 else "right",
            x=1.05 if len(subjects_to_plot) > 6 else 1
        ),
        template='plotly_white'
    )
    
    return fig

def create_single_student_pdf(grades_df, subjects_to_plot, student_name="", class_name=""):
    """创建单个学生的PDF图表，在数据点上显示数值"""
    if grades_df.empty or not subjects_to_plot:
        return None
    
    pdf_buffer = BytesIO()
    
    # 在创建PDF前确保中文字体已设置
    setup_chinese_font()
    
    with PdfPages(pdf_buffer) as pdf:
        # 准备数据
        chart_data = grades_df[['考试场次'] + subjects_to_plot].copy()
        
        # 转换为数值类型
        for subject in subjects_to_plot:
            if subject in chart_data.columns:
                chart_data[subject] = pd.to_numeric(chart_data[subject], errors='coerce')
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4尺寸
        
        # 定义不同的标记符号和线型组合，用于黑白打印
        markers = ['o', '^', 's', 'D', 'v', '*', 'p', 'h', '8', 'H']
        line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10))]
        
        x = range(len(chart_data['考试场次']))
        x_labels = chart_data['考试场次'].tolist()
        
        for idx, subject in enumerate(subjects_to_plot):
            if subject in chart_data.columns:
                marker_idx = idx % len(markers)
                line_idx = idx % len(line_styles)
                
                y = chart_data[subject].values
                ax.plot(x, y, 
                       marker=markers[marker_idx], 
                       linestyle=line_styles[line_idx],
                       linewidth=1.5, 
                       markersize=6, 
                       label=subject)
                
                # 在数据点上添加数值标签
                for i, (xi, yi) in enumerate(zip(x, y)):
                    if not np.isnan(yi):
                        # 根据数值大小调整标签位置
                        offset = 2 if yi >= 0 else -2
                        ax.text(xi, yi + offset, f'{yi:.1f}', 
                               ha='center', va='bottom' if yi >= 0 else 'top',
                               fontsize=9, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.2', 
                                        facecolor='white', 
                                        alpha=0.7, 
                                        edgecolor='lightgray'))
        
        # 设置x轴标签
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
        
        # 设置标题
        title = f"{student_name} 成绩趋势图" if student_name else "成绩趋势图"
        if class_name:
            title = f"{class_name} - {title}"
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('考试场次', fontsize=12)
        ax.set_ylabel('成绩', fontsize=12)
        
        # 添加图例
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=10)
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 自动调整布局
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # 保存到PDF
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def create_pdf_with_charts(student_grades_dict, subjects_to_plot, charts_per_page=6):
    """创建包含多个学生图表的PDF文件，在数据点上显示数值"""
    pdf_buffer = BytesIO()
    
    # 在创建PDF前确保中文字体已设置
    setup_chinese_font()
    
    with PdfPages(pdf_buffer) as pdf:
        # 获取所有学生
        student_keys = list(student_grades_dict.keys())
        total_students = len(student_keys)
        
        # 计算需要的页数
        pages = (total_students + charts_per_page - 1) // charts_per_page
        
        for page in range(pages):
            # 计算当前页的学生索引范围
            start_idx = page * charts_per_page
            end_idx = min(start_idx + charts_per_page, total_students)
            current_students = student_keys[start_idx:end_idx]
            
            # 根据每页图表数量确定布局
            if charts_per_page == 4:
                rows, cols = 2, 2
            elif charts_per_page == 6:
                rows, cols = 2, 3
            else:  # charts_per_page == 8
                rows, cols = 2, 4
            
            # 创建图形
            fig, axes = plt.subplots(rows, cols, figsize=(11.69, 8.27))  # A4尺寸
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
            # 定义不同的标记符号和线型组合，用于黑白打印
            markers = ['o', '^', 's', 'D', 'v', '*', 'p', 'h', '8', 'H']
            line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10))]
            
            # 为当前页的每个学生创建图表
            for idx, student_key in enumerate(current_students):
                ax = axes[idx] if idx < len(axes) else None
                
                if ax is not None:
                    # 获取学生信息
                    parts = student_key.split('_', 1)
                    if len(parts) == 2:
                        class_name, student_name = parts
                    else:
                        class_name, student_name = "未知", student_key
                    
                    # 获取学生成绩数据
                    student_grades_df = student_grades_dict.get(student_key)
                    if student_grades_df is not None and subjects_to_plot:
                        # 准备数据
                        chart_data = student_grades_df[['考试场次'] + subjects_to_plot].copy()
                        
                        # 转换为数值类型
                        for subject in subjects_to_plot:
                            if subject in chart_data.columns:
                                chart_data[subject] = pd.to_numeric(chart_data[subject], errors='coerce')
                        
                        x = range(len(chart_data['考试场次']))
                        x_labels = chart_data['考试场次'].tolist()
                        
                        for subj_idx, subject in enumerate(subjects_to_plot):
                            if subject in chart_data.columns:
                                marker_idx = subj_idx % len(markers)
                                line_idx = subj_idx % len(line_styles)
                                
                                y = chart_data[subject].values
                                ax.plot(x, y, 
                                       marker=markers[marker_idx], 
                                       linestyle=line_styles[line_idx],
                                       linewidth=1, 
                                       markersize=3, 
                                       label=subject)
                                
                                # 在数据点上添加数值标签（只显示前3个和后3个数据点，避免过于密集）
                                for i, (xi, yi) in enumerate(zip(x, y)):
                                    if not np.isnan(yi):
                                        # 只显示关键数据点
                                        if i < 3 or i >= len(x) - 3 or i % 2 == 0:
                                            offset = 1.5 if yi >= 0 else -1.5
                                            ax.text(xi, yi + offset, f'{yi:.0f}', 
                                                   ha='center', va='bottom' if yi >= 0 else 'top',
                                                   fontsize=5, fontweight='bold',
                                                   bbox=dict(boxstyle='round,pad=0.1', 
                                                            facecolor='white', 
                                                            alpha=0.7, 
                                                            edgecolor='lightgray'))
                        
                        # 设置x轴标签
                        ax.set_xticks(x)
                        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=6)
                        
                        # 设置标题
                        title = f"{class_name} - {student_name}"
                        ax.set_title(title, fontsize=8, fontweight='bold', pad=3)
                        ax.set_xlabel('考试场次', fontsize=7)
                        ax.set_ylabel('成绩', fontsize=7)
                        
                        # 设置网格
                        ax.grid(True, alpha=0.3, linestyle='--')
                        
                        # 添加图例
                        if len(subjects_to_plot) <= 5:  # 科目较少时显示图例
                            ax.legend(fontsize=6, loc='upper right')
            
            # 隐藏多余的子图
            for idx in range(len(current_students), len(axes)):
                if idx < len(axes):
                    axes[idx].axis('off')
            
            # 设置总标题
            fig.suptitle(f'学生成绩趋势图 (第{page+1}/{pages}页)', fontsize=12, fontweight='bold', y=0.98)
            
            # 调整布局
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
            
            # 保存当前页到PDF
            pdf.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def update_chart_subjects():
    """更新图表科目的回调函数"""
    st.session_state.chart_updated = True

def get_chart_html_download_link(fig, filename, text):
    """生成HTML格式的图表下载链接"""
    # 将图表转换为HTML
    html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    # 创建完整的HTML文档
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{filename.replace('.html', '')}</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # 编码为base64
    b64 = base64.b64encode(full_html.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{text}</a>'
    return href

def get_chart_data_download_link(grades_df, filename, text):
    """生成图表数据的CSV下载链接"""
    csv_data = grades_df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:text/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def create_charts_zip_html(student_charts):
    """创建包含所有图表HTML文件的ZIP文件"""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for student_key, fig in student_charts.items():
            if fig:
                parts = student_key.split('_', 1)
                if len(parts) == 2:
                    class_name, student_name = parts
                else:
                    class_name, student_name = "未知", student_key
                
                # 生成HTML内容
                html_content = fig.to_html(full_html=True, include_plotlyjs='cdn')
                
                # 创建完整的HTML文档
                full_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>{class_name}_{student_name}_成绩趋势图</title>
                </head>
                <body>
                    <h1>{class_name} - {student_name} 成绩趋势图</h1>
                    {html_content}
                </body>
                </html>
                """
                
                # 添加到ZIP文件
                filename = f"{class_name}_{student_name}_成绩趋势图.html"
                zip_file.writestr(filename, full_html.encode('utf-8'))
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def convert_to_excel(df):
    """将DataFrame转换为Excel字节流"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='学生成绩', index=False)
    return output.getvalue()

def convert_batch_to_excel(df):
    """将批量查询结果转换为Excel字节流"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='批量查询成绩', index=False)
    return output.getvalue()

def update_batch_global_subjects():
    """更新批量查询全局科目的回调函数"""
    st.session_state.batch_charts_generated = False
    st.session_state.show_batch_charts = False
    st.session_state.batch_subjects_modified = True

def generate_all_batch_charts():
    """为所有学生生成图表"""
    st.session_state.batch_charts_generated = True
    st.session_state.show_batch_charts = True
    st.session_state.batch_student_charts = {}
    
    # 为每个学生生成图表
    for student_key, student_grades_df in st.session_state.batch_student_grades.items():
        if st.session_state.batch_global_subjects:
            # 过滤可用的科目
            available_subjects = [s for s in st.session_state.batch_global_subjects 
                                if s in student_grades_df.columns]
            if available_subjects:
                parts = student_key.split('_', 1)
                if len(parts) == 2:
                    class_name, student_name = parts
                else:
                    class_name, student_name = "未知", student_key
                
                fig = create_grade_trend_chart(student_grades_df, available_subjects, student_name, class_name)
                if fig:
                    st.session_state.batch_student_charts[student_key] = fig

# ------------------------------
# 主应用界面
# ------------------------------
def main():
    # 文件上传部分
    st.markdown("---")
    st.markdown("### 📁 上传辅助文件")
    
    uploaded_file = st.file_uploader(
        "选择Excel文件（支持.xlsx, .xls格式）",
        type=["xlsx", "xls"],
        help="请上传包含学生成绩的Excel文件"
    )
    
    if uploaded_file is not None:
        # 显示文件信息
        file_info = st.empty()
        file_info.info(f"📄 已上传文件: {uploaded_file.name}")
        
        # 如果文件已加载，跳过重新加载
        if not st.session_state.data_loaded or st.session_state.df is None:
            with st.spinner("正在加载数据，请稍候..."):
                df = load_data(uploaded_file)
            
            if df is not None:
                # 检测列名
                column_names = detect_column_names(df)
                
                # 检查必要列
                if 'class' not in column_names:
                    st.error("❌ 无法识别班级列，请确保文件包含班级信息")
                    st.info("尝试检查列名是否包含：班别、班级、班、Class等")
                    st.stop()
                
                if 'name' not in column_names:
                    st.error("❌ 无法识别姓名列，请确保文件包含学生姓名信息")
                    st.info("尝试检查列名是否包含：姓名、Name、学生姓名等")
                    st.stop()
                
                # 保存列名到会话状态
                st.session_state.class_column_name = column_names.get('class', '班别')
                st.session_state.name_column_name = column_names.get('name', '姓名')
                st.session_state.id_column_name = column_names.get('id', '学籍号')
                
                # 显示数据概览
                st.success(f"✅ 数据加载成功！共 {len(df):,} 名学生，{len(df.columns)} 个数据列")
                st.info(f"识别到的列名：班级列='{st.session_state.class_column_name}', 姓名列='{st.session_state.name_column_name}'")
                
                # 提取科目、考试场次和列名映射
                with st.spinner("正在解析列名结构..."):
                    info_columns = [
                        st.session_state.class_column_name,
                        st.session_state.name_column_name,
                        st.session_state.id_column_name
                    ]
                    subjects, exams, column_mapping = extract_subjects_exams(df.columns, info_columns)
                
                # 保存到会话状态
                st.session_state.df = df
                st.session_state.subjects = subjects
                st.session_state.exams = exams
                st.session_state.column_mapping = column_mapping
                st.session_state.data_loaded = True
                
                # 初始化默认可视化科目
                default_subjects = ['语文', '数学', '外语'][:min(3, len(subjects))]
                st.session_state.selected_viz_subjects = default_subjects
        else:
            df = st.session_state.df
        
        if st.session_state.data_loaded:
            # 显示识别结果
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("学生总数", f"{len(st.session_state.df):,}")
            with info_col2:
                st.metric("识别科目数", len(st.session_state.subjects))
            with info_col3:
                st.metric("考试场次数", len(st.session_state.exams))
            
            # 显示列名信息
            st.info(f"**使用的列名**：班级列='{st.session_state.class_column_name}', 姓名列='{st.session_state.name_column_name}'")
            
            # 显示详细解析结果
            with st.expander("📊 数据解析详情"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**识别到的科目:**")
                    for i, subject in enumerate(st.session_state.subjects, 1):
                        st.write(f"{i}. {subject}")
                
                with col2:
                    st.markdown("**识别到的考试场次:**")
                    for i, exam in enumerate(st.session_state.exams, 1):
                        st.write(f"{i}. {exam}")
                
                # 显示列名映射示例
                st.markdown("**列名映射示例（前10个）:**")
                mapping_df = pd.DataFrame([
                    {"原始列名": col, "科目": subj, "考试场次": exam} 
                    for col, (subj, exam) in list(st.session_state.column_mapping.items())[:10]
                ])
                st.dataframe(mapping_df, use_container_width=True)
            
            # 查询界面
            st.markdown("---")
            st.markdown("### 🔍 学生成绩查询")
            
            # 获取班级列表
            classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
            
            if not classes:
                st.error("未找到班级信息，请确保班级列包含有效数据")
                st.stop()
            
            # 创建查询列
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                selected_class = st.selectbox(
                    "选择班级",
                    classes,
                    help="从下拉列表中选择班级，支持搜索"
                )
            
            with col2:
                # 根据选择的班级筛选学生
                if selected_class:
                    class_students = st.session_state.df[
                        st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == selected_class
                    ][st.session_state.name_column_name].dropna().unique()
                    
                    if len(class_students) > 0:
                        selected_student = st.selectbox(
                            "选择学生",
                            sorted(class_students),
                            help="从下拉列表中选择学生姓名"
                        )
                    else:
                        st.warning("该班级没有学生数据")
                        selected_student = None
                else:
                    selected_student = None
            
            with col3:
                st.markdown(" ")  # 占位
                st.markdown(" ")  # 占位
                query_clicked = st.button("🔍 查询", type="primary", use_container_width=True)
            
            # 执行查询
            if query_clicked and selected_student:
                with st.spinner(f"正在查询 {selected_class} - {selected_student} 的成绩..."):
                    # 获取学生成绩
                    grades_df = get_student_grades(
                        st.session_state.df, selected_class, selected_student,
                        st.session_state.class_column_name, st.session_state.name_column_name,
                        st.session_state.subjects, st.session_state.exams, 
                        st.session_state.column_mapping
                    )
                    
                    if grades_df is not None and not grades_df.empty:
                        # 保存到会话状态
                        st.session_state.grades_df = grades_df
                        st.session_state.current_student = f"{selected_class} - {selected_student}"
                        st.session_state.chart_updated = True
                        st.session_state.single_pdf_created = False
                        st.session_state.single_pdf_data = None
                        
                        # 显示学生基本信息
                        st.markdown("---")
                        
                        # 获取学生学籍号
                        student_info = st.session_state.df[
                            (st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == selected_class) & 
                            (st.session_state.df[st.session_state.name_column_name] == selected_student)
                        ].iloc[0]
                        
                        student_id = ""
                        if st.session_state.id_column_name in student_info:
                            student_id = str(student_info[st.session_state.id_column_name])
                        
                        # 信息卡片
                        st.markdown(f"### 🎓 {selected_class} - {selected_student} 的成绩记录")
                        
                        info_cols = st.columns(4)
                        with info_cols[0]:
                            st.metric("班级", selected_class)
                        with info_cols[1]:
                            st.metric("姓名", selected_student)
                        with info_cols[2]:
                            if student_id:
                                st.metric("学籍号", student_id)
                        with info_cols[3]:
                            exam_count = len(grades_df)
                            st.metric("考试场次", exam_count)
                        
                        # 显示成绩表格
                        st.markdown("#### 📊 各科成绩汇总")
                        
                        # 格式化显示
                        display_df = grades_df.copy()
                        display_df = display_df.set_index('考试场次')
                        
                        # 对数值列进行格式化
                        for col in display_df.columns:
                            if display_df[col].dtype in ['int64', 'float64']:
                                if '排' in col:
                                    display_df[col] = display_df[col].apply(
                                        lambda x: f"{int(x)}" if pd.notna(x) and not np.isnan(x) else "-"
                                    )
                                else:
                                    display_df[col] = display_df[col].apply(
                                        lambda x: f"{x:.1f}" if pd.notna(x) and not np.isnan(x) else "-"
                                    )
                            else:
                                display_df[col] = display_df[col].apply(
                                    lambda x: str(x) if pd.notna(x) else "-"
                                )
                        
                        # 显示表格
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            height=min(600, 100 + len(display_df) * 35)
                        )
                        
                        # 数据统计
                        st.markdown("#### 📈 成绩统计")
                        
                        # 计算各科目平均分
                        score_subjects = [s for s in st.session_state.subjects if '排' not in s]
                        
                        if score_subjects:
                            stats_cols = st.columns(min(4, len(score_subjects)))
                            
                            for idx, subject in enumerate(score_subjects[:4]):
                                with stats_cols[idx]:
                                    if subject in grades_df.columns:
                                        col_data = pd.to_numeric(grades_df[subject], errors='coerce')
                                        valid_data = col_data.dropna()
                                        
                                        if len(valid_data) > 0:
                                            avg_score = valid_data.mean()
                                            max_score = valid_data.max()
                                            min_score = valid_data.min()
                                            
                                            st.metric(
                                                f"{subject}平均分",
                                                f"{avg_score:.1f}",
                                                f"最高: {max_score:.1f} | 最低: {min_score:.1f}"
                                            )
                                        else:
                                            st.metric(f"{subject}平均分", "-", "无有效数据")
                        
                        # 成绩趋势可视化
                        st.markdown("#### 📈 成绩趋势图")
                        st.markdown("选择要可视化的科目：")
                        
                        with st.form(key="chart_form"):
                            # 显示排名开关
                            show_rankings = st.checkbox(
                                "显示排名科目（如三排、总排等）",
                                value=st.session_state.get('show_rankings', False),
                                key="show_rankings_checkbox"
                            )
                            st.session_state.show_rankings = show_rankings
                            
                            # 获取可用的科目
                            if show_rankings:
                                # 显示所有科目，包括排名
                                available_subjects = [s for s in st.session_state.subjects 
                                                    if s in st.session_state.grades_df.columns]
                            else:
                                # 只显示成绩科目，不显示排名
                                available_subjects = [s for s in st.session_state.subjects 
                                                    if s in st.session_state.grades_df.columns and '排' not in s]
                            
                            if not st.session_state.selected_viz_subjects:
                                # 如果之前选择的科目包含排名，但现在不显示排名，则过滤掉
                                if not show_rankings:
                                    default_subjects = [s for s in ['语文', '数学', '外语'] 
                                                      if s in available_subjects][:min(3, len(available_subjects))]
                                else:
                                    default_subjects = ['语文', '数学', '外语'][:min(3, len(available_subjects))]
                                st.session_state.selected_viz_subjects = default_subjects
                            
                            # 过滤已选择的科目，确保它们都在可用科目列表中
                            current_selected = [s for s in st.session_state.selected_viz_subjects if s in available_subjects]
                            
                            selected_subjects = st.multiselect(
                                "科目选择",
                                available_subjects,
                                default=current_selected,
                                label_visibility="collapsed"
                            )
                            
                            submit_button = st.form_submit_button("更新图表", on_click=update_chart_subjects)
                        
                        if st.session_state.chart_updated and selected_subjects:
                            fig = create_grade_trend_chart(st.session_state.grades_df, selected_subjects, selected_student, selected_class)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                st.session_state.selected_viz_subjects = selected_subjects
                        
                        # 下载功能
                        st.markdown("#### 💾 数据导出")
                        
                        excel_data = convert_to_excel(st.session_state.grades_df)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.download_button(
                                label="📥 下载Excel格式",
                                data=excel_data,
                                file_name=f"{selected_class}_{selected_student}_成绩表_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        with col2:
                            csv_df = st.session_state.grades_df.copy()
                            csv_data = csv_df.to_csv(index=False, encoding='utf-8-sig')
                            
                            st.download_button(
                                label="📥 下载CSV格式",
                                data=csv_data,
                                file_name=f"{selected_class}_{selected_student}_成绩表_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        
                        with col3:
                            # 单个学生PDF下载
                            if selected_subjects:
                                # 创建PDF按钮
                                create_pdf_clicked = st.button("📄 创建PDF图表", 
                                                              type="secondary", 
                                                              use_container_width=True,
                                                              key="create_single_pdf_button")
                                
                                if create_pdf_clicked:
                                    with st.spinner("正在创建PDF图表..."):
                                        pdf_data = create_single_student_pdf(
                                            st.session_state.grades_df, 
                                            selected_subjects, 
                                            selected_student, 
                                            selected_class
                                        )
                                        if pdf_data:
                                            st.session_state.single_pdf_data = pdf_data
                                            st.session_state.single_pdf_created = True
                                            st.success("✅ PDF图表已创建！")
                                
                                if st.session_state.single_pdf_created and st.session_state.single_pdf_data:
                                    st.download_button(
                                        label="📥 下载PDF图表",
                                        data=st.session_state.single_pdf_data,
                                        file_name=f"{selected_class}_{selected_student}_成绩趋势图_{datetime.now().strftime('%Y%m%d')}.pdf",
                                        mime="application/pdf"
                                    )
                            else:
                                st.info("请先选择要可视化的科目")
                    else:
                        st.error(f"❌ 未找到学生 {selected_class} - {selected_student} 的成绩数据，或数据为空")
            elif st.session_state.grades_df is not None and st.session_state.current_student:
                # 如果已有查询结果，显示历史结果
                st.info(f"📊 当前显示的是上一次查询结果: {st.session_state.current_student}")
                
                display_df = st.session_state.grades_df.copy().set_index('考试场次')
                
                for col in display_df.columns:
                    if display_df[col].dtype in ['int64', 'float64']:
                        if '排' in col:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{int(x)}" if pd.notna(x) and not np.isnan(x) else "-"
                            )
                        else:
                            display_df[col] = display_df[col].apply(
                                lambda x: f"{x:.1f}" if pd.notna(x) and not np.isnan(x) else "-"
                            )
                    else:
                        display_df[col] = display_df[col].apply(
                            lambda x: str(x) if pd.notna(x) else "-"
                        )
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=min(600, 100 + len(display_df) * 35)
                )
                
                st.markdown("#### 📈 成绩趋势图")
                st.markdown("选择要可视化的科目：")
                
                with st.form(key="chart_form_existing"):
                    # 显示排名开关
                    show_rankings = st.checkbox(
                        "显示排名科目（如三排、总排等）",
                        value=st.session_state.get('show_rankings', False),
                        key="show_rankings_checkbox_existing"
                    )
                    st.session_state.show_rankings = show_rankings
                    
                    # 获取可用的科目
                    if show_rankings:
                        # 显示所有科目，包括排名
                        available_subjects = [s for s in st.session_state.subjects 
                                            if s in st.session_state.grades_df.columns]
                    else:
                        # 只显示成绩科目，不显示排名
                        available_subjects = [s for s in st.session_state.subjects 
                                            if s in st.session_state.grades_df.columns and '排' not in s]
                    
                    # 过滤已选择的科目，确保它们都在可用科目列表中
                    current_selected = [s for s in st.session_state.selected_viz_subjects if s in available_subjects]
                    
                    selected_subjects = st.multiselect(
                        "科目选择",
                        available_subjects,
                        default=current_selected,
                        label_visibility="collapsed"
                    )
                    
                    submit_button = st.form_submit_button("更新图表", on_click=update_chart_subjects)
                
                if st.session_state.chart_updated and selected_subjects:
                    fig = create_grade_trend_chart(st.session_state.grades_df, selected_subjects)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.session_state.selected_viz_subjects = selected_subjects
                
                # 单个学生PDF下载
                st.markdown("#### 💾 PDF导出")
                
                if selected_subjects:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 创建PDF按钮
                        create_pdf_clicked = st.button("📄 创建PDF图表", 
                                                      type="secondary", 
                                                      use_container_width=True,
                                                      key="create_single_pdf_button_existing")
                        
                        if create_pdf_clicked:
                            with st.spinner("正在创建PDF图表..."):
                                pdf_data = create_single_student_pdf(
                                    st.session_state.grades_df, 
                                    selected_subjects, 
                                    st.session_state.current_student
                                )
                                if pdf_data:
                                    st.session_state.single_pdf_data = pdf_data
                                    st.session_state.single_pdf_created = True
                                    st.success("✅ PDF图表已创建！")
                    
                    with col2:
                        if st.session_state.single_pdf_created and st.session_state.single_pdf_data:
                            st.download_button(
                                label="📥 下载PDF图表",
                                data=st.session_state.single_pdf_data,
                                file_name=f"{st.session_state.current_student}_成绩趋势图_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                else:
                    st.info("请先选择要可视化的科目")
            else:
                if query_clicked:
                    st.warning("请选择班级和学生")
            
            # 批量查询功能
            st.markdown("---")
            st.markdown("### 📋 批量查询功能")
            
            st.markdown("""
            **使用说明**：
            1. 选择查询模式：手动输入或分班查询
            2. 如果选择手动输入模式：
               - 在文本框中输入要查询的学生信息
               - 每行输入一个学生，格式为：`班级,姓名`
            3. 如果选择分班查询模式：
               - 从下拉列表中选择要查询的班级
               - 可以多选多个班级
            4. 点击"执行批量查询"按钮
            5. 系统将查询所有学生的成绩并分别展示
            6. 选择要查看的科目，然后点击"一键生成所有学生图表"按钮
            7. 系统会为所有学生一次性生成成绩趋势图
            
            **注意**：请确保输入的班级和姓名与数据中的完全一致
            """)
            
            # 显示可用班级和学生示例
            with st.expander("👀 查看可用班级和学生示例"):
                if len(classes) > 0:
                    sample_class = classes[0]
                    sample_students = st.session_state.df[
                        st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == sample_class
                    ][st.session_state.name_column_name].dropna().unique()[:5]
                    
                    st.markdown(f"**示例班级:** {sample_class}")
                    st.markdown(f"**该班级前5名学生:** {', '.join(sample_students)}")
                    st.markdown(f"**输入示例:**")
                    st.code(f"{sample_class},{sample_students[0] if len(sample_students) > 0 else '学生姓名'}")
            
            # 查询模式选择
            st.markdown("#### 📊 选择查询模式")
            query_mode = st.radio(
                "查询模式",
                ["手动输入模式（逐行输入）", "分班查询模式（查询整个班级）"],
                index=0 if st.session_state.batch_query_mode == "manual" else 1,
                horizontal=True,
                key="batch_query_mode_selector"
            )
            
            # 更新查询模式
            if "手动" in query_mode:
                st.session_state.batch_query_mode = "manual"
            else:
                st.session_state.batch_query_mode = "class_bulk"
            
            batch_input = ""
            batch_data = []
            
            if st.session_state.batch_query_mode == "manual":
                # 手动输入模式
                batch_input = st.text_area(
                    "输入批量查询内容（每行一个学生，格式：班级,姓名）",
                    height=150,
                    placeholder=f"例如：\n{classes[0] if len(classes) > 0 else '1'},覃楚静\n{classes[0] if len(classes) > 0 else '1'},黄和梅\n{classes[1] if len(classes) > 1 else '2'},王五",
                    help="支持一次查询多个学生，每行一个。注意：使用半角逗号分隔"
                )
                
                # 解析批量查询输入
                if batch_input.strip():
                    lines = [line.strip() for line in batch_input.split('\n') if line.strip()]
                    
                    for line in lines:
                        if ',' in line or '，' in line:
                            line_clean = line.replace('，', ',')
                            parts = [part.strip() for part in line_clean.split(',', 1)]
                            if len(parts) == 2:
                                batch_class, batch_name = parts
                                batch_data.append((batch_class, batch_name))
            else:
                # 分班查询模式
                st.markdown("#### 🏫 选择要查询的班级")
                selected_classes = st.multiselect(
                    "选择班级（可多选）",
                    classes,
                    default=st.session_state.selected_batch_classes,
                    help="选择要查询的班级，可以多选多个班级"
                )
                
                # 更新会话状态
                st.session_state.selected_batch_classes = selected_classes
                
                if selected_classes:
                    # 显示选中的班级和学生数量
                    st.info(f"已选择 {len(selected_classes)} 个班级")
                    
                    for class_name in selected_classes:
                        class_student_count = len(st.session_state.df[
                            st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == class_name
                        ][st.session_state.name_column_name].dropna().unique())
                        
                        st.write(f"- **{class_name}**: {class_student_count} 名学生")
                    
                    # 获取所有选中的班级的学生
                    batch_data = get_class_all_students(
                        st.session_state.df, selected_classes,
                        st.session_state.class_column_name, st.session_state.name_column_name
                    )
                    
                    st.success(f"✅ 已准备查询 {len(batch_data)} 名学生")
            
            batch_query_clicked = st.button("🔍 执行批量查询", type="secondary", key="batch_query_button")
            
            if batch_query_clicked:
                st.session_state.batch_query_executed = True
            
            if st.session_state.batch_query_executed and ((st.session_state.batch_query_mode == "manual" and batch_input.strip()) or (st.session_state.batch_query_mode == "class_bulk" and batch_data)):
                if batch_data:
                    with st.spinner(f"正在批量查询 {len(batch_data)} 名学生..."):
                        batch_results, found_students, not_found_students, student_grades_dict = get_batch_student_grades(
                            st.session_state.df, batch_data, 
                            st.session_state.class_column_name, st.session_state.name_column_name,
                            st.session_state.subjects, st.session_state.exams, 
                            st.session_state.column_mapping
                        )
                        
                        if batch_results is not None and not batch_results.empty:
                            st.success(f"✅ 批量查询完成！找到 {len(found_students)} 名学生，{len(batch_results)} 条成绩记录")
                            
                            # 保存到会话状态
                            st.session_state.batch_results = batch_results
                            st.session_state.batch_student_grades = student_grades_dict
                            st.session_state.batch_charts_generated = False
                            st.session_state.show_batch_charts = False
                            
                            # 重置全局科目选择
                            all_available_subjects = []
                            for student_key, student_grades_df in student_grades_dict.items():
                                student_subjects = [s for s in st.session_state.subjects 
                                                  if s in student_grades_df.columns]
                                all_available_subjects.extend(student_subjects)
                            
                            unique_subjects = list(set(all_available_subjects))
                            default_subjects = ['语文', '数学', '外语'][:min(3, len(unique_subjects))]
                            st.session_state.batch_global_subjects = default_subjects
                            
                            if found_students:
                                st.markdown(f"**✅ 找到的学生 ({len(found_students)}名):**")
                                
                                # 按班级分组显示
                                class_groups = {}
                                for class_name, student_name in found_students:
                                    if class_name not in class_groups:
                                        class_groups[class_name] = []
                                    class_groups[class_name].append(student_name)
                                
                                for class_name, students in class_groups.items():
                                    with st.expander(f"**{class_name} ({len(students)}名)**"):
                                        for i, student_name in enumerate(sorted(students), 1):
                                            st.write(f"{i}. {student_name}")
                            
                            if not_found_students:
                                st.warning(f"**❌ 未找到的学生 ({len(not_found_students)}名):**")
                                for i, (class_name, student_name) in enumerate(not_found_students, 1):
                                    st.write(f"{i}. {class_name} - {student_name}")
                                st.info("请检查班级和姓名是否与数据中的完全一致")
                            
                            # 显示批量查询结果表格
                            st.markdown("#### 📊 批量查询结果")
                            
                            batch_display_df = batch_results.copy()
                            
                            for col in batch_display_df.columns:
                                if col not in ['班级', '姓名', '考试场次']:
                                    if batch_display_df[col].dtype in ['int64', 'float64']:
                                        if '排' in col:
                                            batch_display_df[col] = batch_display_df[col].apply(
                                                lambda x: f"{int(x)}" if pd.notna(x) and not np.isnan(x) else "-"
                                            )
                                        else:
                                            batch_display_df[col] = batch_display_df[col].apply(
                                                lambda x: f"{x:.1f}" if pd.notna(x) and not np.isnan(x) else "-"
                                            )
                                    else:
                                        batch_display_df[col] = batch_display_df[col].apply(
                                            lambda x: str(x) if pd.notna(x) else "-"
                                        )
                            
                            st.dataframe(
                                batch_display_df,
                                use_container_width=True,
                                height=min(600, 200 + len(batch_display_df) * 35)
                            )
                            
                            # 批量图表生成功能
                            st.markdown("#### 📈 批量成绩趋势图生成")
                            
                            # 获取所有可用的科目
                            all_available_subjects = []
                            for student_key, student_grades_df in student_grades_dict.items():
                                student_subjects = [s for s in st.session_state.subjects 
                                                  if s in student_grades_df.columns]
                                all_available_subjects.extend(student_subjects)
                            
                            unique_subjects = sorted(list(set(all_available_subjects)))
                            
                            if unique_subjects:
                                st.markdown("##### 选择要为所有学生生成的科目：")
                                
                                # 显示排名开关
                                show_rankings = st.checkbox(
                                    "显示排名科目（如三排、总排等）",
                                    value=st.session_state.get('batch_show_rankings', False),
                                    key="batch_show_rankings_checkbox"
                                )
                                st.session_state.batch_show_rankings = show_rankings
                                
                                # 根据开关过滤科目
                                if show_rankings:
                                    # 显示所有科目，包括排名
                                    filtered_subjects = unique_subjects
                                else:
                                    # 只显示成绩科目，不显示排名
                                    filtered_subjects = [s for s in unique_subjects if '排' not in s]
                                
                                # 科目选择器
                                selected_global_subjects = st.multiselect(
                                    "科目选择",
                                    filtered_subjects,
                                    default=[s for s in st.session_state.batch_global_subjects if s in filtered_subjects],
                                    key="batch_global_subjects_selector"
                                )
                                
                                # 直接更新会话状态
                                st.session_state.batch_global_subjects = selected_global_subjects
                                
                                # 每页图表数量选择
                                st.markdown("##### 选择每页显示的图表数量：")
                                charts_per_page = st.selectbox(
                                    "每页图表数",
                                    [4, 6, 8],
                                    index=1,  # 默认选择6
                                    key="charts_per_page_selector"
                                )
                                
                                # 保存每页图表数量到会话状态
                                st.session_state.charts_per_page_value = charts_per_page
                                
                                # 生成图表按钮
                                generate_charts_clicked = st.button("🚀 一键生成所有学生图表", 
                                                                   type="primary", 
                                                                   use_container_width=True,
                                                                   key="generate_batch_charts_button")
                                
                                if generate_charts_clicked:
                                    if not st.session_state.batch_global_subjects:
                                        st.warning("⚠️ 请先选择要生成的科目")
                                    else:
                                        with st.spinner("正在生成所有学生图表..."):
                                            generate_all_batch_charts()
                                
                                if st.session_state.batch_charts_generated:
                                    st.success("✅ 所有学生图表已生成！")
                            
                            else:
                                st.info("没有找到可用的成绩科目数据")
                            
                            # 显示图表
                            if st.session_state.show_batch_charts and st.session_state.batch_student_charts:
                                st.markdown("---")
                                st.markdown("#### 📊 各学生成绩趋势图")
                                
                                student_charts = st.session_state.batch_student_charts
                                
                                for idx, (student_key, fig) in enumerate(student_charts.items(), 1):
                                    if fig:
                                        parts = student_key.split('_', 1)
                                        if len(parts) == 2:
                                            class_name, student_name = parts
                                        
                                        st.markdown(f"##### 🎓 {class_name} - {student_name}")
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # 提供下载链接
                                        st.markdown("**图表下载选项：**")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            html_link = get_chart_html_download_link(
                                                fig,
                                                f"{class_name}_{student_name}_成绩趋势图.html",
                                                "📥 下载HTML图表"
                                            )
                                            st.markdown(html_link, unsafe_allow_html=True)
                                        
                                        with col2:
                                            if student_key in st.session_state.batch_student_grades:
                                                student_grades_df = st.session_state.batch_student_grades[student_key]
                                                csv_link = get_chart_data_download_link(
                                                    student_grades_df,
                                                    f"{class_name}_{student_name}_成绩数据.csv",
                                                    "📥 下载数据CSV"
                                                )
                                                st.markdown(csv_link, unsafe_allow_html=True)
                                        
                                        st.markdown("---")
                            
                            # 批量下载功能
                            st.markdown("#### 💾 批量查询结果导出")
                            
                            batch_excel_data = convert_batch_to_excel(batch_results)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.download_button(
                                    label="📥 下载合并Excel",
                                    data=batch_excel_data,
                                    file_name=f"批量查询_成绩表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            
                            with col2:
                                batch_csv_data = batch_results.to_csv(index=False, encoding='utf-8-sig')
                                
                                st.download_button(
                                    label="📥 下载合并CSV",
                                    data=batch_csv_data,
                                    file_name=f"批量查询_成绩表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                            
                            with col3:
                                # 批量下载所有图表为HTML
                                if st.session_state.batch_student_charts:
                                    zip_html_data = create_charts_zip_html(st.session_state.batch_student_charts)
                                    st.download_button(
                                        label="📦 下载所有HTML图表",
                                        data=zip_html_data,
                                        file_name=f"批量查询_成绩趋势图_HTML_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                        mime="application/zip"
                                    )
                            
                            with col4:
                                # 批量下载所有图表为PDF
                                if st.session_state.batch_student_grades and st.session_state.batch_global_subjects:
                                    # 创建PDF
                                    pdf_data = create_pdf_with_charts(
                                        st.session_state.batch_student_grades,
                                        st.session_state.batch_global_subjects,
                                        st.session_state.charts_per_page_value
                                    )
                                    
                                    st.download_button(
                                        label="📄 下载合并PDF",
                                        data=pdf_data,
                                        file_name=f"批量查询_成绩趋势图_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf"
                                    )
                        else:
                            st.error("❌ 批量查询未找到任何匹配的学生数据")
                            
                            if not_found_students:
                                st.warning(f"**未找到的学生列表 ({len(not_found_students)}名):**")
                                for i, (class_name, student_name) in enumerate(not_found_students, 1):
                                    st.write(f"{i}. {class_name} - {student_name}")
                                
                                st.info(f"""
                                **可能的原因：**
                                1. 班级名称不匹配（注意：班级列名是'{st.session_state.class_column_name}'）
                                2. 学生姓名不匹配
                                3. 输入格式不正确
                                
                                **建议：**
                                1. 使用上方的"查看可用班级和学生示例"功能
                                2. 复制示例中的班级和学生姓名
                                3. 检查输入格式是否正确（班级,姓名）
                                """)
                else:
                    st.warning("⚠️ 请输入有效的批量查询内容（每行格式：班级,姓名）")
            else:
                if st.session_state.batch_query_executed and not batch_input.strip():
                    st.warning("⚠️ 请输入批量查询内容")
            
            # 如果之前有批量查询结果，也显示出来
            if st.session_state.batch_results is not None:
                st.markdown("---")
                st.markdown("### 📊 历史批量查询结果")
                
                batch_display_df = st.session_state.batch_results.copy()
                
                for col in batch_display_df.columns:
                    if col not in ['班级', '姓名', '考试场次']:
                        if batch_display_df[col].dtype in ['int64', 'float64']:
                            if '排' in col:
                                batch_display_df[col] = batch_display_df[col].apply(
                                    lambda x: f"{int(x)}" if pd.notna(x) and not np.isnan(x) else "-"
                                )
                            else:
                                batch_display_df[col] = batch_display_df[col].apply(
                                    lambda x: f"{x:.1f}" if pd.notna(x) and not np.isnan(x) else "-"
                                )
                        else:
                            batch_display_df[col] = batch_display_df[col].apply(
                                lambda x: str(x) if pd.notna(x) else "-"
                            )
                
                st.dataframe(
                    batch_display_df,
                    use_container_width=True,
                    height=min(400, 100 + len(batch_display_df) * 35)
                )
                
                # 如果之前有生成图表，也显示出来
                if st.session_state.batch_student_charts:
                    st.markdown("---")
                    st.markdown("### 📈 历史批量查询图表")
                    
                    student_charts = st.session_state.batch_student_charts
                    
                    for idx, (student_key, fig) in enumerate(student_charts.items(), 1):
                        if fig:
                            parts = student_key.split('_', 1)
                            if len(parts) == 2:
                                class_name, student_name = parts
                            
                            st.markdown(f"##### 🎓 {class_name} - {student_name}")
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown("---")
    else:
        # 上传文件前的提示
        st.markdown("---")
        
        st.info("""
        ### 📋 使用说明
        
        1. **准备数据文件**
           - Excel文件需要包含班级和学生姓名信息
           - 成绩列命名格式：`科目` + `考试场次`
        
        2. **上传文件**
           - 点击"浏览文件"按钮或拖拽文件到上传区域
           - 系统会自动解析列名结构
        
        3. **查询成绩**
           - 从下拉列表选择班级
           - 从下拉列表选择学生姓名
           - 点击"查询"按钮查看成绩
        
        4. **批量查询**
           - 在批量查询区域输入多行数据
           - 每行格式：`班级,姓名`
           - 点击"执行批量查询"按钮
           - 选择要查看的科目
           - 点击"一键生成所有学生图表"按钮
           - 系统会一次性为所有学生生成成绩趋势图
        
        5. **功能特性**
           - 支持大规模数据（千名学生）
           - 自动识别科目和考试场次
           - 提供成绩趋势可视化图表
           - 支持多种格式导出（Excel、CSV、HTML、PDF）
           - 支持批量查询和导出
           - **数值标签**：图表数据点上显示具体数值
           - **PDF优化**：黑白打印友好，使用不同标记和线型区分科目
           - **单个学生PDF**：单个学生查询也可导出PDF图表
           - **中文字体支持**：PDF中的中文可以正常显示
           - **优化交互**：下载按钮已移出表单，可以正常使用
        
        ### ⚠️ 注意事项
        
        - 确保Excel文件格式正确
        - 班级和姓名需与数据中的完全一致
        - 支持.xlsx和.xls格式文件
        """)
        
        st.markdown("---")
        st.caption("💡 提示：首次使用请确保Excel文件格式正确，系统将自动识别科目和考试场次")

# 运行应用
if __name__ == "__main__":
    main()


