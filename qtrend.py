import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import base64
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import zipfile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
from matplotlib import font_manager
import os
import sys
import warnings
import textwrap
from typing import List, Dict, Tuple, Optional, Any

warnings.filterwarnings('ignore') #无视警告

# 导入必要的库
import plotly.express as px
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import defaultdict
import math
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import hashlib

# 尝试导入statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.api import SimpleExpSmoothing
    statsmodels_available = True
except ImportError:
    statsmodels_available = False
    st.warning("注意：statsmodels库未安装，将使用替代方法。")

# ============================================
# 配置类
# ============================================
class Config:
    """系统配置"""
    # 不同操作系统的中文字体配置
    FONT_DIRS = [
        '/usr/share/fonts',
        '/usr/local/share/fonts',
        'C:/Windows/Fonts',
        '/System/Library/Fonts',
        '/Library/Fonts',
    ]
    
    CHINESE_FONTS = [
        'simhei.ttf', 'simsun.ttc', 'msyh.ttc',
        'STKAITI.TTF', 'STSONG.TTF', 'DroidSansFallback.ttf',
    ]
    
    # 已知科目
    KNOWN_SUBJECTS = [
        '语文', '数学', '外语', '政治', '历史', '地理',
        '物理', '化学', '生物', '三总', '三排', '总分', '总排'
    ]
    
    # 科目颜色
    SUBJECT_COLORS = {
        '语文': '#1f77b4',
        '数学': '#ff7f0e',
        '外语': '#2ca02c',
        '政治': '#d62728',
        '历史': '#9467bd',
        '地理': '#8c564b',
        '物理': '#e377c2',
        '化学': '#7f7f7f',
        '生物': '#bcbd22',
        '三总': '#17becf',
        '总分': '#393b79',
    }
    
    # 图表默认配置
    CHART_HEIGHT = 500
    CHART_TEMPLATE = 'plotly_white'
    
    # PDF配置
    PDF_PAGE_SIZE = (11.69, 8.27)  # PDF格式设置为A4
    PDF_DPI = 300
    
    # 缓存配置
    CACHE_TTL = 3600  # 计算结果保存1小时，1小时内相同查询不用重新计算

# ============================================
# 字体管理类
# ============================================
class FontManager:
    """字体管理器"""
    
    @staticmethod
    def setup_chinese_font():
        """设置中文字体支持"""
        try:
            # 尝试查找可用的中文字体
            found_font = FontManager._find_chinese_font()
            
            if found_font:
                # 添加字体到matplotlib
                font_manager.fontManager.addfont(found_font)
                font_name = font_manager.FontProperties(fname=found_font).get_name()
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False #解决负号显示为方框的问题
                return True
            else:
                # 使用默认字体
                FontManager._setup_default_font()
                return True
        except Exception as e:
            print(f"设置中文字体时出错: {e}", file=sys.stderr)
            return False
    
    @staticmethod
    def _find_chinese_font():
        """查找中文字体"""
        for font_dir in Config.FONT_DIRS:
            if os.path.exists(font_dir):
                for font_file in Config.CHINESE_FONTS:
                    font_path = os.path.join(font_dir, font_file)
                    if os.path.exists(font_path):
                        return font_path
        return None
    
    @staticmethod
    def _setup_default_font():
        """设置默认字体"""
        try:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

# ============================================
# 考试场次智能排序器
# ============================================
class ExamSorter:
    """考试场次智能排序器"""
    
    @staticmethod
    def parse_exam_name(exam_name: str) -> Dict[str, Any]:
        """
        解析考试场次名称
        格式: 年级学期+考试类型/月份
        例如: 一二期中, 三一十二月
        返回解析后的字典
        """
        if not exam_name or len(exam_name) < 3:
            return {
                'grade': 99,  # 默认值，表示无法解析
                'semester': 99,
                'exam_type': '',
                'month': 0,
                'parsed': False
            }
        
        # 解析年级和学期
        # 第一个字符是年级: 一(高一), 二(高二), 三(高三)
        # 第二个字符是学期: 一(第一学期), 二(第二学期)
        grade_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6}
        semester_map = {'一': 1, '二': 2, '上': 1, '下': 2}
        
        grade_char = exam_name[0]
        semester_char = exam_name[1] if len(exam_name) > 1 else ''
        
        grade = grade_map.get(grade_char, 99)
        semester = semester_map.get(semester_char, 99)
        
        # 提取剩余部分作为考试描述
        exam_desc = exam_name[2:] if len(exam_name) > 2 else ''
        
        # 定义考试类型优先级
        exam_type_order = {
            '期中': 1, '五校': 2, '八校': 3, '月考': 4,
            '联考': 5, '期末': 6
        }
        
        # 尝试匹配考试类型
        exam_type = ''
        exam_priority = 99
        month = 0
        
        for exam_type_name, priority in exam_type_order.items():
            if exam_type_name in exam_desc:
                exam_type = exam_type_name
                exam_priority = priority
                break
        
        # 如果没有匹配到已知考试类型，检查是否是月考（带月份）
        if not exam_type and '月' in exam_desc:
            exam_type = '月考'
            exam_priority = exam_type_order.get('月考', 9)
            
            # 提取月份
            month_match = re.search(r'(\d{1,2})月|([一二三四五六七八九十]+)月', exam_desc)
            if month_match:
                if month_match.group(1):  # 数字月份
                    month = int(month_match.group(1))
                else:  # 中文月份
                    chinese_month = month_match.group(2)
                    chinese_month_map = {
                        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6,
                        '七': 7, '八': 8, '九': 9, '十': 10, '十一': 11, '十二': 12
                    }
                    month = chinese_month_map.get(chinese_month, 0)
        
        return {
            'grade': grade,
            'semester': semester,
            'exam_type': exam_type,
            'exam_priority': exam_priority,
            'month': month,
            'original': exam_name,
            'parsed': grade != 99 and semester != 99
        }
    
    @staticmethod
    def sort_exams(exam_list: List[str]) -> List[str]:
        """
        对考试场次列表进行智能排序
        排序规则:
        1. 按年级升序 (一, 二, 三)
        2. 按学期升序 (一, 二)
        3. 按考试类型优先级 (期中, 月考, 模拟, 五校, 八校, 联考, 联考, 期末)
        4. 按月考试月份升序
        5. 按原始字符串排序
        """
        if not exam_list:
            return []
        
        # 解析所有考试场次
        parsed_exams = []
        for exam in exam_list:
            parsed = ExamSorter.parse_exam_name(exam)
            parsed_exams.append(parsed)
        
        # 定义排序键
        def sort_key(parsed_exam):
            return (
                parsed_exam['grade'],           # 年级
                parsed_exam['semester'],        # 学期
                parsed_exam['exam_priority'],   # 考试类型优先级
                parsed_exam['month'],           # 月份（月考）
                parsed_exam['original']         # 原始字符串
            )
        
        # 排序
        sorted_parsed = sorted(parsed_exams, key=sort_key)
        
        # 返回排序后的原始字符串
        return [exam['original'] for exam in sorted_parsed]
    
    @staticmethod
    def get_exam_details(exam_name: str) -> Dict[str, Any]:
        """
        获取考试场次的详细信息
        """
        parsed = ExamSorter.parse_exam_name(exam_name)
        
        if parsed['parsed']:
            grade_names = {1: '高一', 2: '高二', 3: '高三'}
            semester_names = {1: '第一学期', 2: '第二学期'}
            
            return {
                'original_name': exam_name,
                'grade': parsed['grade'],
                'grade_name': grade_names.get(parsed['grade'], f'未知年级({parsed["grade"]})'),
                'semester': parsed['semester'],
                'semester_name': semester_names.get(parsed['semester'], f'未知学期({parsed["semester"]})'),
                'exam_type': parsed['exam_type'],
                'exam_priority': parsed['exam_priority'],
                'month': parsed['month'],
                'description': ExamSorter._generate_description(parsed)
            }
        else:
            return {
                'original_name': exam_name,
                'error': '无法解析考试场次名称',
                'description': f'无法解析: {exam_name}'
            }
    
    @staticmethod
    def _generate_description(parsed_exam: Dict[str, Any]) -> str:
        """生成考试场次描述"""
        grade_names = {1: '高一', 2: '高二', 3: '高三'}
        semester_names = {1: '第一学期', 2: '第二学期'}
        
        grade = grade_names.get(parsed_exam['grade'], f'{parsed_exam["grade"]}年级')
        semester = semester_names.get(parsed_exam['semester'], f'{parsed_exam["semester"]}学期')
        
        if parsed_exam['exam_type'] == '月考' and parsed_exam['month'] > 0:
            return f'{grade}{semester}{parsed_exam["month"]}月月考'
        elif parsed_exam['exam_type']:
            return f'{grade}{semester}{parsed_exam["exam_type"]}考试'
        else:
            return parsed_exam['original']

# ============================================
# 数据处理器
# ============================================
class DataProcessor:
    """数据处理类"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def load_data(uploaded_file):
        """加载并缓存Excel数据"""
        try:
            # 尝试不同的引擎
            try:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            except:
                try:
                    df = pd.read_excel(uploaded_file, engine='xlrd')
                except:
                    df = pd.read_excel(uploaded_file)
            
            # 清理列名
            df.columns = df.columns.astype(str).str.strip()
            return df
        except Exception as e:
            st.error(f"文件读取失败: {str(e)}")
            return None
    
    @staticmethod
    def detect_column_names(df):
        """智能检测列名"""
        column_names = {}
        df_columns = [str(col).strip() for col in df.columns]
        
        # 检测班级列
        class_patterns = ['班别', '班级', '班', 'Class', 'class', 'CLS', 'cls']
        column_names['class'] = DataProcessor._find_column(df_columns, class_patterns)
        
        # 检测姓名列
        name_patterns = ['姓名', 'Name', 'name', '学生姓名', '学生名']
        column_names['name'] = DataProcessor._find_column(df_columns, name_patterns)
        
        # 检测学号列
        id_patterns = ['学籍号', '学号', 'ID', 'id', 'StudentID', 'student_id']
        column_names['id'] = DataProcessor._find_column(df_columns, id_patterns)
        
        return column_names
    
    @staticmethod
    def _find_column(columns, patterns):
        """查找匹配的列"""
        for col in columns:
            for pattern in patterns:
                if pattern in col:
                    return col
        return None
    
    @staticmethod
    def extract_subjects_exams(df_columns, info_columns):
        """从列名中智能提取科目和考试场次"""
        # 基础信息列
        base_columns = [str(col).strip() for col in info_columns if col]
        
        # 提取所有非基础列
        grade_columns = [col for col in df_columns if col not in base_columns]
        
        if not grade_columns:
            return [], [], {}
        
        subjects = set()
        column_mapping = {}
        
        # 先尝试精确匹配已知科目
        for col in grade_columns:
            matched = False
            for subject in Config.KNOWN_SUBJECTS:
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
        sorted_subjects = []
        for priority in Config.KNOWN_SUBJECTS:
            if priority in subjects:
                sorted_subjects.append(priority)
                subjects.discard(priority)
        
        # 添加剩余的科目
        sorted_subjects.extend(sorted(subjects))
        
        # 提取所有考试场次
        exams = set()
        for _, exam in column_mapping.values():
            exams.add(exam)
        
        # 对考试场次进行智能排序
        sorted_exams = ExamSorter.sort_exams(list(exams))
        
        return sorted_subjects, sorted_exams, column_mapping

# ============================================
# 新增：考试权重计算器
# ============================================
class ExamWeightCalculator:
    """考试重要性权重计算器"""
    
    def __init__(self):
        # 考试类型权重
        self.exam_type_weights = {
            '期末': 1.0, '期中': 0.5, '月考': 0.9, '模拟': 0.5,
            '联考': 0.5, '五校': 0.5, '八校': 0.5, '质检': 0.5
        }
        self.time_decay_rate = 0.15  # 时间衰减率
        self.recent_weight_boost = 0.2  # 近期考试额外权重
        
    def calculate_exam_weight(self, exam_name: str, exam_index: int, 
                            total_exams: int, is_recent: bool = False) -> float:
        """
        计算考试权重
        Args:
            exam_name: 考试名称
            exam_index: 考试序号（0表示最近一次）
            total_exams: 总考试次数
            is_recent: 是否为近期考试
        Returns:
            权重值
        """
        # 基础权重
        base_weight = 0.5
        for exam_type, weight in self.exam_type_weights.items():
            if exam_type in str(exam_name):
                base_weight = weight
                break
        
        # 时间衰减权重（最近考试权重更高）
        time_weight = 1.0 - (exam_index / max(total_exams, 1)) * self.time_decay_rate
        
        # 近期考试额外权重
        recent_weight = 1.0 + (self.recent_weight_boost if is_recent else 0.0)
        
        # 综合权重
        final_weight = base_weight * time_weight * recent_weight
        
        return min(final_weight, 1.2)  # 设置上限

# ============================================
# 成绩趋势分析器（增强版）
# ============================================
class EnhancedGradeTrendAnalyzer:
    """增强版成绩趋势分析器"""
    
    def __init__(self):
        self.weight_calculator = ExamWeightCalculator()
        
    def calculate_trend_stats(self, grades: np.ndarray, exam_names: List[str], 
                             weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        计算成绩趋势统计
        Args:
            grades: 成绩数组
            exam_names: 考试名称列表
            weights: 权重数组
        Returns:
            趋势统计字典
        """
        if len(grades) < 2:
            return {"error": "数据不足"}
        
        # 计算考试权重
        if weights is None:
            weights = []
            for i, exam_name in enumerate(exam_names):
                is_recent = (i >= len(exam_names) - 3)  # 最近3次考试
                weight = self.weight_calculator.calculate_exam_weight(
                    exam_name, i, len(exam_names), is_recent
                )
                weights.append(weight)
        
        # 加权线性回归
        x = np.arange(len(grades))
        y = np.array(grades)
        w = np.array(weights)
        
        A = np.vstack([x * w, w]).T
        b = y * w
        coeffs = np.linalg.lstsq(A, b, rcond=None)[0]
        slope, intercept = coeffs[0], coeffs[1]
        
        # 预测下一次考试
        next_x = len(grades)
        next_grade = slope * next_x + intercept
        
        # 计算趋势
        if slope > 0.5:
            trend = "上升趋势"
            trend_level = "strong_up"
        elif slope > 0.1:
            trend = "轻微上升"
            trend_level = "weak_up"
        elif slope < -0.5:
            trend = "下降趋势"
            trend_level = "strong_down"
        elif slope < -0.1:
            trend = "轻微下降"
            trend_level = "weak_down"
        else:
            trend = "平稳"
            trend_level = "stable"
        
        # 计算稳定性
        if len(grades) > 1:
            stability = np.std(grades) / np.mean(grades) if np.mean(grades) > 0 else 0
        else:
            stability = 0
        
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "next_grade": float(next_grade),
            "trend": trend,
            "trend_level": trend_level,
            "stability": float(stability),
            "current_grade": float(grades[-1]),
            "mean_grade": float(np.mean(grades)),
            "exam_weights": weights
        }

# ============================================
# 成绩管理器
# ============================================
class GradeManager:
    """成绩管理类"""
    
    @staticmethod
    def get_student_grades(df, class_name, student_name, class_col, name_col, 
                          subjects, exams, column_mapping):
        """获取指定学生的成绩数据"""
        try:
            # 确保班级名称类型一致
            df_class_col = df[class_col].astype(str).str.strip()
            input_class_name = str(class_name).strip()
            
            # 筛选学生数据
            mask = (df_class_col == input_class_name) & (df[name_col] == student_name)
            student_data = df[mask]
            
            if student_data.empty:
                return None
            
            student_row = student_data.iloc[0]
            return GradeManager._build_student_grades_df(student_row, subjects, exams, column_mapping)
        
        except Exception as e:
            st.error(f"获取学生成绩时出错: {str(e)}")
            return None
    
    @staticmethod
    def get_batch_student_grades(df, batch_data, class_col, name_col, 
                                subjects, exams, column_mapping):
        """批量获取多个学生的成绩数据"""
        all_results = []
        found_students = []
        not_found_students = []
        student_grades_dict = {}
        
        for class_name, student_name in batch_data:
            try:
                grades_df = GradeManager.get_student_grades(
                    df, class_name, student_name, class_col, name_col,
                    subjects, exams, column_mapping
                )
                
                if grades_df is not None and not grades_df.empty:
                    found_students.append((class_name, student_name))
                    student_grades_dict[f"{class_name}_{student_name}"] = grades_df
                    
                    # 添加到合并结果
                    for idx, row in grades_df.iterrows():
                        result_row = {'班级': class_name, '姓名': student_name}
                        result_row.update(row.to_dict())
                        all_results.append(result_row)
                else:
                    not_found_students.append((class_name, student_name))
            
            except Exception as e:
                st.warning(f"处理学生 {class_name} - {student_name} 时出错: {str(e)}")
                not_found_students.append((class_name, student_name))
        
        if all_results:
            return pd.DataFrame(all_results), found_students, not_found_students, student_grades_dict
        else:
            return None, [], not_found_students, {}
    
    @staticmethod
    def _build_student_grades_df(student_row, subjects, exams, column_mapping):
        """构建学生成绩DataFrame"""
        result_data = []
        
        for exam in exams:
            row = {'考试场次': exam}
            
            for subject in subjects:
                col_name = GradeManager._find_column_name(subject, exam, column_mapping)
                
                if col_name and col_name in student_row:
                    value = student_row[col_name]
                    row[subject] = GradeManager._format_value(value)
                else:
                    row[subject] = None
            
            result_data.append(row)
        
        return pd.DataFrame(result_data)
    
    @staticmethod
    def _find_column_name(subject, exam, column_mapping):
        """查找列名"""
        for col, (subj, exm) in column_mapping.items():
            if subj == subject and exm == exam:
                return col
        return None
    
    @staticmethod
    def _format_value(value):
        """格式化值"""
        if pd.isna(value):
            return None
        
        try:
            # 尝试转换为数值
            if isinstance(value, (int, float)):
                return float(value)
            
            # 处理字符串形式的数值
            if isinstance(value, str):
                value = value.strip()
                if value == '' or value.lower() in ['null', 'nan', 'none']:
                    return None
                
                # 尝试转换为数值
                try:
                    return float(value)
                except ValueError:
                    # 如果是排名，可能包含特殊字符
                    if '排' in str(value):
                        try:
                            return int(re.sub(r'[^\d]', '', value))
                        except:
                            return str(value)
                    return str(value)
            
            return float(value)
        
        except (ValueError, TypeError):
            return str(value)
    
    @staticmethod
    def get_class_all_students(df, class_names, class_col, name_col):
        """获取指定班级的所有学生名单"""
        batch_data = []
        
        for class_name in class_names:
            try:
                df_class_col = df[class_col].astype(str).str.strip()
                input_class_name = str(class_name).strip()
                
                mask = (df_class_col == input_class_name)
                class_students = df[mask][name_col].dropna().unique()
                
                for student_name in class_students:
                    batch_data.append((class_name, student_name))
            
            except Exception as e:
                st.warning(f"获取班级 {class_name} 学生名单时出错: {str(e)}")
        
        return batch_data

# ============================================
# 图表生成器
# ============================================
class ChartGenerator:
    """图表生成器"""
    
    @staticmethod
    def create_grade_trend_chart(grades_df, subjects_to_plot, 
                                student_name="", class_name="", 
                                show_values=True, height=None):
        """创建成绩趋势图表（Plotly版本）"""
        if grades_df.empty or not subjects_to_plot:
            return None
        
        try:
            # 准备数据
            chart_data = grades_df[['考试场次'] + subjects_to_plot].copy()
            
            # 转换为数值类型
            for subject in subjects_to_plot:
                if subject in chart_data.columns:
                    chart_data[subject] = pd.to_numeric(chart_data[subject], errors='coerce')
            
            # 创建图表
            fig = go.Figure()
            
            # 为每个科目添加一条线
            for idx, subject in enumerate(subjects_to_plot):
                if subject in chart_data.columns:
                    color = Config.SUBJECT_COLORS.get(
                        subject, 
                        f'hsl({(idx * 137) % 360}, 70%, 50%)'  # 生成区分度高的颜色
                    )
                    
                    y_values = chart_data[subject].values
                    
                    # 创建trace
                    trace = go.Scatter(
                        x=chart_data['考试场次'],
                        y=y_values,
                        mode='lines+markers',
                        name=subject,
                        line=dict(color=color, width=3),
                        marker=dict(size=8, color=color),
                        hovertemplate=(
                            f'<b>{subject}</b><br>' +
                            '考试场次: %{x}<br>' +
                            '成绩: %{y:.1f}<br>' +
                            '<extra></extra>'
                        )
                    )
                    
                    fig.add_trace(trace)
                    
                    # 如果需要显示数值标签
                    if show_values:
                        for i, (x_val, y_val) in enumerate(zip(chart_data['考试场次'], y_values)):
                            if not np.isnan(y_val):
                                fig.add_annotation(
                                    x=x_val,
                                    y=y_val,
                                    text=f'{y_val:.1f}',
                                    showarrow=False,
                                    yshift=10,
                                    font=dict(size=10, color=color),
                                    bgcolor='rgba(255, 255, 255, 0.8)',
                                    bordercolor=color,
                                    borderwidth=1,
                                    borderpad=2
                                )
            
            # 更新图表布局
            title = f"{class_name} - {student_name} 成绩趋势图" if class_name and student_name else "成绩趋势图"
            if not class_name and student_name:
                title = f"{student_name} 成绩趋势图"
            
            if height is None:
                height = Config.CHART_HEIGHT + (len(subjects_to_plot) - 3) * 30
            
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=20, family="Arial, sans-serif")
                ),
                xaxis_title='考试场次',
                yaxis_title='成绩',
                height=height,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02,
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='lightgray',
                    borderwidth=1
                ),
                template=Config.CHART_TEMPLATE,
                margin=dict(l=50, r=100, t=80, b=50),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(
                    gridcolor='lightgray',
                    showgrid=True,
                    gridwidth=1
                ),
                yaxis=dict(
                    gridcolor='lightgray',
                    showgrid=True,
                    gridwidth=1
                )
            )
            
            return fig
        
        except Exception as e:
            st.error(f"创建图表时出错: {str(e)}")
            return None
    
    @staticmethod
    def create_comparison_chart(grades_df_list, student_names, subjects_to_plot, 
                               class_name="", height=600):
        """创建多学生对比图表"""
        if not grades_df_list or not subjects_to_plot:
            return None
        
        try:
            fig = go.Figure()
            
            # 为每个学生和每个科目创建trace
            for student_idx, (grades_df, student_name) in enumerate(zip(grades_df_list, student_names)):
                for subject_idx, subject in enumerate(subjects_to_plot):
                    if subject in grades_df.columns:
                        color_idx = student_idx * len(subjects_to_plot) + subject_idx
                        color = f'hsl({(color_idx * 137) % 360}, 70%, 50%)'
                        
                        y_values = pd.to_numeric(grades_df[subject], errors='coerce').values
                        
                        # 创建trace
                        trace_name = f"{student_name} - {subject}"
                        trace = go.Scatter(
                            x=grades_df['考试场次'],
                            y=y_values,
                            mode='lines+markers',
                            name=trace_name,
                            line=dict(color=color, width=2, dash='solid' if student_idx == 0 else 'dash'),
                            marker=dict(size=6, color=color, symbol='circle' if student_idx == 0 else 'square'),
                            hovertemplate=(
                                f'<b>{student_name}</b><br>' +
                                f'<b>{subject}</b><br>' +
                                '考试场次: %{x}<br>' +
                                '成绩: %{y:.1f}<br>' +
                                '<extra></extra>'
                            )
                        )
                        
                        fig.add_trace(trace)
            
            # 更新布局
            title = f"{class_name} 学生成绩对比图" if class_name else "学生成绩对比图"
            
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=20)
                ),
                xaxis_title='考试场次',
                yaxis_title='成绩',
                height=height,
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02
                ),
                template=Config.CHART_TEMPLATE
            )
            
            return fig
        
        except Exception as e:
            st.error(f"创建对比图表时出错: {str(e)}")
            return None

# ============================================
# PDF生成器
# ============================================
class PDFGenerator:
    """PDF生成器"""
    
    @staticmethod
    def create_single_student_pdf(grades_df, subjects_to_plot, student_name="", 
                                 class_name="", filename=None):
        """创建单个学生的PDF图表"""
        if grades_df.empty or not subjects_to_plot:
            return None
        
        try:
            # 确保中文字体
            FontManager.setup_chinese_font()
            
            pdf_buffer = BytesIO()
            
            with PdfPages(pdf_buffer) as pdf:
                # 准备数据
                chart_data = grades_df[['考试场次'] + subjects_to_plot].copy()
                
                # 转换为数值类型
                for subject in subjects_to_plot:
                    if subject in chart_data.columns:
                        chart_data[subject] = pd.to_numeric(chart_data[subject], errors='coerce')
                
                # 创建图表
                fig, ax = plt.subplots(figsize=Config.PDF_PAGE_SIZE)
                
                # 定义不同的标记符号和线型组合
                markers = ['o', '^', 's', 'D', 'v', '*', 'p', 'h', '8', 'H', '<', '>']
                line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10))]
                
                x = range(len(chart_data['考试场次']))
                x_labels = [str(label) for label in chart_data['考试场次'].tolist()]
                
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
                title = f"{class_name} - {student_name} 成绩趋势图" if class_name and student_name else "成绩趋势图"
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel('考试场次', fontsize=12)
                ax.set_ylabel('成绩', fontsize=12)
                
                # 添加图例
                ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=10)
                
                # 设置网格
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # 调整布局
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                
                # 保存到PDF
                pdf.savefig(fig, dpi=Config.PDF_DPI, bbox_inches='tight')
                plt.close(fig)
            
            pdf_buffer.seek(0)
            return pdf_buffer.getvalue()
        
        except Exception as e:
            st.error(f"创建PDF时出错: {str(e)}")
            return None
    
    @staticmethod
    def create_pdf_with_charts(student_grades_dict, subjects_to_plot, 
                              charts_per_page=6, title="学生成绩趋势图"):
        """创建包含多个学生图表的PDF文件"""
        try:
            FontManager.setup_chinese_font()
            
            pdf_buffer = BytesIO()
            
            with PdfPages(pdf_buffer) as pdf:
                student_keys = list(student_grades_dict.keys())
                total_students = len(student_keys)
                
                if total_students == 0:
                    return None
                
                # 计算需要的页数
                pages = (total_students + charts_per_page - 1) // charts_per_page
                
                for page in range(pages):
                    # 计算当前页的学生索引范围
                    start_idx = page * charts_per_page
                    end_idx = min(start_idx + charts_per_page, total_students)
                    current_students = student_keys[start_idx:end_idx]
                    
                    # 确定布局
                    if charts_per_page == 1:
                        rows, cols = 1, 1
                    elif charts_per_page == 2:
                        rows, cols = 1, 2
                    elif charts_per_page == 4:
                        rows, cols = 2, 2
                    elif charts_per_page == 6:
                        rows, cols = 2, 3
                    elif charts_per_page == 8:
                        rows, cols = 2, 4
                    elif charts_per_page == 9:
                        rows, cols = 3, 3
                    else:
                        rows, cols = 3, 4
                    
                    # 创建图形
                    fig, axes = plt.subplots(rows, cols, figsize=Config.PDF_PAGE_SIZE)
                    
                    if rows == 1 and cols == 1:
                        axes = [[axes]]
                    elif rows == 1:
                        axes = [axes]
                    elif cols == 1:
                        axes = [[ax] for ax in axes]
                    
                    axes_flat = [ax for row in axes for ax in row]
                    
                    # 定义不同的标记符号和线型组合
                    markers = ['o', '^', 's', 'D', 'v', '*', 'p', 'h', '8', 'H']
                    line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10))]
                    
                    # 为当前页的每个学生创建图表
                    for idx, student_key in enumerate(current_students):
                        if idx < len(axes_flat):
                            ax = axes_flat[idx]
                            
                            # 获取学生信息
                            parts = student_key.split('_', 1)
                            if len(parts) == 2:
                                class_name, student_name = parts
                            else:
                                class_name, student_name = "未知", student_key
                            
                            # 获取学生成绩数据
                            student_grades_df = student_grades_dict.get(student_key)
                            if student_grades_df is not None and subjects_to_plot:
                                PDFGenerator._plot_student_chart(
                                    ax, student_grades_df, subjects_to_plot,
                                    class_name, student_name,
                                    markers, line_styles
                                )
                    
                    # 隐藏多余的子图
                    for idx in range(len(current_students), len(axes_flat)):
                        if idx < len(axes_flat):
                            axes_flat[idx].axis('off')
                    
                    # 设置总标题
                    if pages > 1:
                        page_title = f'{title} (第{page+1}/{pages}页)'
                    else:
                        page_title = title
                    
                    fig.suptitle(page_title, fontsize=16, fontweight='bold', y=0.98)
                    
                    # 调整布局
                    plt.tight_layout(rect=[0, 0, 1, 0.96])
                    
                    # 保存当前页到PDF
                    pdf.savefig(fig, dpi=Config.PDF_DPI, bbox_inches='tight')
                    plt.close(fig)
            
            pdf_buffer.seek(0)
            return pdf_buffer.getvalue()
        
        except Exception as e:
            st.error(f"创建批量PDF时出错: {str(e)}")
            return None
    
    @staticmethod
    def _plot_student_chart(ax, student_grades_df, subjects_to_plot,
                           class_name, student_name, markers, line_styles):
        """绘制单个学生图表"""
        try:
            # 准备数据
            chart_data = student_grades_df[['考试场次'] + subjects_to_plot].copy()
            
            # 转换为数值类型
            for subject in subjects_to_plot:
                if subject in chart_data.columns:
                    chart_data[subject] = pd.to_numeric(chart_data[subject], errors='coerce')
            
            x = range(len(chart_data['考试场次']))
            x_labels = [str(label) for label in chart_data['考试场次'].tolist()]
            
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
            
            # 设置x轴标签
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=6)
            
            # 设置标题
            title = f"{class_name} - {student_name}"
            if len(title) > 20:
                title = textwrap.fill(title, 20)
            ax.set_title(title, fontsize=8, fontweight='bold', pad=3)
            ax.set_xlabel('考试场次', fontsize=7)
            ax.set_ylabel('成绩', fontsize=7)
            
            # 设置网格
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # 添加图例
            if len(subjects_to_plot) <= 5:
                ax.legend(fontsize=6, loc='upper right')
        
        except Exception as e:
            st.warning(f"绘制学生 {student_name} 图表时出错: {str(e)}")

# ============================================
# 会话状态管理器
# ============================================
class SessionManager:
    """会话状态管理器"""
    
    @staticmethod
    def init_session_state():
        """初始化会话状态"""
        # 数据状态
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
        
        # 查询状态
        if 'selected_viz_subjects' not in st.session_state:
            st.session_state.selected_viz_subjects = []
        if 'grades_df' not in st.session_state:
            st.session_state.grades_df = None
        if 'current_student' not in st.session_state:
            st.session_state.current_student = None
        if 'chart_updated' not in st.session_state:
            st.session_state.chart_updated = True
        
        # 批量查询状态
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
        if 'batch_query_mode' not in st.session_state:
            st.session_state.batch_query_mode = "manual"
        if 'selected_batch_classes' not in st.session_state:
            st.session_state.selected_batch_classes = []
        
        # 显示状态
        if 'show_rankings' not in st.session_state:
            st.session_state.show_rankings = False
        if 'batch_show_rankings' not in st.session_state:
            st.session_state.batch_show_rankings = False
        
        # 导出状态
        if 'charts_per_page_value' not in st.session_state:
            st.session_state.charts_per_page_value = 6
        if 'single_pdf_created' not in st.session_state:
            st.session_state.single_pdf_created = False
        if 'single_pdf_data' not in st.session_state:
            st.session_state.single_pdf_data = None
        
        # 新增：分析状态
        if 'analysis_type' not in st.session_state:
            st.session_state.analysis_type = 'trend'
        if 'selected_analysis_subject' not in st.session_state:
            st.session_state.selected_analysis_subject = None
        if 'comparison_students' not in st.session_state:
            st.session_state.comparison_students = []

# ============================================
# 导出工具类
# ============================================
class ExportTool:
    """数据导出工具类"""
    
    @staticmethod
    def convert_to_excel(df):
        """将DataFrame转换为Excel字节流"""
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='学生成绩', index=False)
            return output.getvalue()
        except Exception as e:
            st.error(f"转换为Excel时出错: {str(e)}")
            return None
    
    @staticmethod
    def convert_batch_to_excel(df):
        """将批量查询结果转换为Excel字节流"""
        try:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='批量查询成绩', index=False)
            return output.getvalue()
        except Exception as e:
            st.error(f"转换为批量Excel时出错: {str(e)}")
            return None
    
    @staticmethod
    def get_chart_html_download_link(fig, filename, text):
        """生成HTML格式的图表下载链接"""
        try:
            # 将图表转换为HTML
            html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
            
            # 创建完整的HTML文档
            full_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{filename.replace('.html', '')}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                </style>
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """
            
            # 编码为base64
            b64 = base64.b64encode(full_html.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="{filename}" class="download-link">{text}</a>'
            return href
        except Exception as e:
            st.error(f"生成HTML下载链接时出错: {str(e)}")
            return ""
    
    @staticmethod
    def get_chart_data_download_link(grades_df, filename, text):
        """生成图表数据的CSV下载链接"""
        try:
            csv_data = grades_df.to_csv(index=False, encoding='utf-8-sig')
            b64 = base64.b64encode(csv_data.encode()).decode()
            href = f'<a href="data:text/csv;base64,{b64}" download="{filename}" class="download-link">{text}</a>'
            return href
        except Exception as e:
            st.error(f"生成CSV下载链接时出错: {str(e)}")
            return ""
    
    @staticmethod
    def create_charts_zip_html(student_charts):
        """创建包含所有图表HTML文件的ZIP文件"""
        try:
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
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                                h1 {{ color: #333; }}
                            </style>
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
        except Exception as e:
            st.error(f"创建ZIP文件时出错: {str(e)}")
            return None

# ============================================
# 回调函数
# ============================================
def update_chart_subjects():
    """更新图表科目的回调函数"""
    st.session_state.chart_updated = True

def update_batch_global_subjects():
    """更新批量查询全局科目的回调函数"""
    st.session_state.batch_charts_generated = False
    st.session_state.show_batch_charts = False
    st.session_state.batch_subjects_modified = True

def generate_all_batch_charts():
    """为所有学生生成图表"""
    try:
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
                    
                    fig = ChartGenerator.create_grade_trend_chart(
                        student_grades_df, available_subjects, student_name, class_name
                    )
                    if fig:
                        st.session_state.batch_student_charts[student_key] = fig
    except Exception as e:
        st.error(f"生成批量图表时出错: {str(e)}")

# ============================================
# 工具函数
# ============================================
def display_data_overview(df, class_col, name_col, subjects, exams):
    """显示数据概览"""
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric("学生总数", f"{len(df):,}")
    with info_col2:
        st.metric("识别科目数", len(subjects))
    with info_col3:
        st.metric("考试场次数", len(exams))
    
    st.info(f"**使用的列名**：班级列='{class_col}', 姓名列='{name_col}'")
    
    with st.expander("📊 数据解析详情"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**识别到的科目:**")
            for i, subject in enumerate(subjects, 1):
                st.write(f"{i}. {subject}")
        
        with col2:
            st.markdown("**识别到的考试场次:**")
            for i, exam in enumerate(exams, 1):
                st.write(f"{i}. {exam}")
        
        # 显示列名映射示例
        if st.session_state.column_mapping:
            st.markdown("**列名映射示例（前10个）:**")
            mapping_data = []
            for col, (subj, exam) in list(st.session_state.column_mapping.items())[:10]:
                mapping_data.append({"原始列名": col, "科目": subj, "考试场次": exam})
            
            if mapping_data:
                mapping_df = pd.DataFrame(mapping_data)
                st.dataframe(mapping_df, use_container_width=True)

def display_student_info(student_data, class_col, name_col, id_col, class_name, student_name):
    """显示学生基本信息"""
    student_id = ""
    if id_col in student_data.columns:
        student_id_val = student_data[id_col].iloc[0]
        if pd.notna(student_id_val):
            student_id = str(student_id_val).strip()
    
    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("班级", class_name)
    with info_cols[1]:
        st.metric("姓名", student_name)
    with info_cols[2]:
        if student_id:
            st.metric("学籍号", student_id)
    with info_cols[3]:
        if st.session_state.grades_df is not None:
            exam_count = len(st.session_state.grades_df)
            st.metric("考试场次", exam_count)
    
    return student_id

def format_grades_dataframe(grades_df):
    """格式化成绩DataFrame用于显示"""
    display_df = grades_df.copy()
    display_df = display_df.set_index('考试场次')
    
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
    
    return display_df

def display_statistics(grades_df, subjects):
    """显示成绩统计信息"""
    score_subjects = [s for s in subjects if '排' not in s and s in grades_df.columns]
    
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

def get_available_subjects(show_rankings, subjects, grades_df):
    """获取可用的科目列表"""
    if show_rankings:
        # 显示所有科目，包括排名
        return [s for s in subjects if s in grades_df.columns]
    else:
        # 只显示成绩科目，不显示排名
        return [s for s in subjects if s in grades_df.columns and '排' not in s]

# ============================================
# 模块1: 导入数据文件
# ============================================
def module_import_data():
    """模块1: 导入数据文件"""
    st.markdown("## 📁 1. 导入数据文件")
    st.markdown("上传包含学生成绩的Excel文件，系统将自动解析文件结构。")
    
    uploaded_file = st.file_uploader(
        "选择Excel文件（支持.xlsx, .xls格式）",
        type=["xlsx", "xls"],
        help="请上传包含学生成绩的Excel文件",
        key="module1_uploader"
    )
    
    if uploaded_file is not None:
        # 显示文件信息
        st.info(f"📄 已上传文件: {uploaded_file.name}")
        
        # 如果文件已加载，跳过重新加载
        if not st.session_state.data_loaded or st.session_state.df is None:
            with st.spinner("正在加载数据，请稍候..."):
                df = DataProcessor.load_data(uploaded_file)
            
            if df is not None:
                # 检测列名
                column_names = DataProcessor.detect_column_names(df)
                
                # 检查必要列
                if 'class' not in column_names:
                    st.error("❌ 无法识别班级列，请确保文件包含班级信息")
                    st.info("尝试检查列名是否包含：班别、班级、班、Class等")
                    return
                
                if 'name' not in column_names:
                    st.error("❌ 无法识别姓名列，请确保文件包含学生姓名信息")
                    st.info("尝试检查列名是否包含：姓名、Name、学生姓名等")
                    return
                
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
                    subjects, exams, column_mapping = DataProcessor.extract_subjects_exams(df.columns, info_columns)
                
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
            # 显示数据概览
            display_data_overview(
                st.session_state.df, 
                st.session_state.class_column_name, 
                st.session_state.name_column_name,
                st.session_state.subjects,
                st.session_state.exams
            )
    else:
        st.info("""
        ### 📋 使用说明
        
        1. **准备数据文件**
           - Excel文件需要包含班级和学生姓名信息
           - 成绩列命名格式：`科目` + `考试场次`
        
        2. **上传文件**
           - 点击"浏览文件"按钮或拖拽文件到上传区域
           - 系统会自动解析列名结构
        
        3. **注意事项**
           - 确保Excel文件格式正确
           - 班级和姓名需与数据中的完全一致
           - 支持.xlsx和.xls格式文件
        """)

# ============================================
# 模块2: 单个学生成绩查询
# ============================================
def module_single_student_query():
    """模块2: 单个学生成绩查询"""
    st.markdown("## 🔍 2. 单个学生成绩查询")
    
    if not st.session_state.data_loaded:
        st.warning("请先上传数据文件（切换到'导入数据文件'模块）")
        return
    
    st.markdown("""
    **使用说明**：
    1. 选择班级和学生
    2. 系统将自动显示该学生的成绩数据
    3. 您可以选择要显示的科目和是否显示排名
    4. 可以下载成绩数据和图表
    """)
    
    # 选择班级和学生
    classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
    
    col1, col2 = st.columns(2)
    with col1:
        selected_class = st.selectbox(
            "选择班级",
            classes,
            key="single_class_select"
        )
    
    with col2:
        if selected_class:
            class_students = st.session_state.df[
                st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == selected_class
            ][st.session_state.name_column_name].dropna().unique()
            
            if len(class_students) > 0:
                selected_student = st.selectbox(
                    "选择学生",
                    sorted(class_students),
                    key="single_student_select"
                )
            else:
                st.warning("该班级没有学生数据")
                selected_student = None
        else:
            selected_student = None
    
    if selected_student:
        # 获取学生成绩
        grades_df = GradeManager.get_student_grades(
            st.session_state.df, selected_class, selected_student,
            st.session_state.class_column_name, st.session_state.name_column_name,
            st.session_state.subjects, st.session_state.exams, 
            st.session_state.column_mapping
        )
        
        if grades_df is not None and not grades_df.empty:
            st.success(f"✅ 成功获取 {selected_class} - {selected_student} 的成绩数据")
            
            # 显示学生基本信息
            student_data = st.session_state.df[
                (st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == selected_class) &
                (st.session_state.df[st.session_state.name_column_name] == selected_student)
            ]
            
            if not student_data.empty:
                student_id = display_student_info(
                    student_data, st.session_state.class_column_name, 
                    st.session_state.name_column_name, st.session_state.id_column_name,
                    selected_class, selected_student
                )
            
            # 显示成绩数据表
            st.markdown("#### 📊 成绩数据表")
            
            # 显示排名开关
            show_rankings = st.checkbox(
                "显示排名科目（如三排、总排等）",
                value=st.session_state.get('show_rankings', False),
                on_change=update_chart_subjects,
                key="show_rankings_checkbox"
            )
            st.session_state.show_rankings = show_rankings
            
            # 获取可用的科目
            available_subjects = get_available_subjects(
                show_rankings, st.session_state.subjects, grades_df
            )
            
            # 科目选择器
            selected_viz_subjects = st.multiselect(
                "选择要显示的科目",
                available_subjects,
                default=[s for s in st.session_state.selected_viz_subjects if s in available_subjects],
                on_change=update_chart_subjects,
                key="viz_subjects_multiselect"
            )
            
            # 更新会话状态
            st.session_state.selected_viz_subjects = selected_viz_subjects
            
            # 显示成绩数据
            display_df = format_grades_dataframe(grades_df)
            
            if not selected_viz_subjects:
                st.info("⚠️ 请至少选择一个科目进行可视化")
                st.dataframe(display_df, use_container_width=True)
            else:
                # 只显示选中的科目
                display_subjects = ['考试场次'] + selected_viz_subjects
                filtered_display_df = display_df[display_subjects] if '考试场次' in display_df.columns else display_df
                st.dataframe(filtered_display_df, use_container_width=True, height=400)
                
                # 显示统计信息
                st.markdown("#### 📈 成绩统计信息")
                display_statistics(grades_df, selected_viz_subjects)
                
                # 创建趋势图表
                st.markdown("#### 📈 成绩趋势图")
                
                # 设置图表高度
                chart_height = Config.CHART_HEIGHT + (len(selected_viz_subjects) - 3) * 30
                
                # 创建图表
                fig = ChartGenerator.create_grade_trend_chart(
                    grades_df, selected_viz_subjects, selected_student, selected_class,
                    show_values=True, height=chart_height
                )
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="single_student_chart")
                    
                    # 图表下载选项
                    st.markdown("#### 💾 下载选项")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # 下载图表为HTML
                        html_link = ExportTool.get_chart_html_download_link(
                            fig,
                            f"{selected_class}_{selected_student}_成绩趋势图.html",
                            "📥 下载HTML图表"
                        )
                        st.markdown(html_link, unsafe_allow_html=True)
                    
                    with col2:
                        # 下载数据为CSV
                        csv_link = ExportTool.get_chart_data_download_link(
                            grades_df,
                            f"{selected_class}_{selected_student}_成绩数据.csv",
                            "📥 下载数据CSV"
                        )
                        st.markdown(csv_link, unsafe_allow_html=True)
                    
                    with col3:
                        # 下载为Excel
                        excel_data = ExportTool.convert_to_excel(grades_df)
                        if excel_data:
                            st.download_button(
                                label="📥 下载Excel",
                                data=excel_data,
                                file_name=f"{selected_class}_{selected_student}_成绩数据.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="single_excel_download"
                            )
                    
                    # PDF导出
                    st.markdown("##### 📄 PDF导出")
                    
                    pdf_col1, pdf_col2, pdf_col3 = st.columns(3)
                    
                    with pdf_col1:
                        if st.button("🖨️ 生成PDF报告", key="generate_pdf_button"):
                            with st.spinner("正在生成PDF报告..."):
                                pdf_data = PDFGenerator.create_single_student_pdf(
                                    grades_df, selected_viz_subjects, 
                                    selected_student, selected_class
                                )
                                
                                if pdf_data:
                                    st.session_state.single_pdf_data = pdf_data
                                    st.session_state.single_pdf_created = True
                                    st.success("PDF报告生成成功！")
                    
                    with pdf_col2:
                        if st.session_state.single_pdf_created and st.session_state.single_pdf_data:
                            st.download_button(
                                label="📥 下载PDF",
                                data=st.session_state.single_pdf_data,
                                file_name=f"{selected_class}_{selected_student}_成绩报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                key="single_pdf_download"
                            )
        else:
            st.error(f"❌ 未找到学生 {selected_class} - {selected_student} 的成绩数据")
    else:
        st.info("👈 请先选择班级和学生")

# ============================================
# 模块3: 批量学生成绩查询
# ============================================
def module_batch_student_query():
    """模块3: 批量学生成绩查询"""
    st.markdown("## 📋 3. 批量学生成绩查询")
    
    if not st.session_state.data_loaded:
        st.warning("请先上传数据文件（切换到'导入数据文件'模块）")
        return
    
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
        classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
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
            placeholder=f"例如：\n{classes[0] if len(classes) > 0 else '1'},学生A\n{classes[0] if len(classes) > 0 else '1'},学生B\n{classes[1] if len(classes) > 1 else '2'},学生C",
            help="支持一次查询多个学生，每行一个。注意：使用半角逗号分隔",
            key="batch_input_area"
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
            help="选择要查询的班级，可以多选多个班级",
            key="batch_class_multiselect"
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
            batch_data = GradeManager.get_class_all_students(
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
                batch_results, found_students, not_found_students, student_grades_dict = GradeManager.get_batch_student_grades(
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
                    
                    batch_display_df = format_grades_dataframe(batch_results)
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
                                st.plotly_chart(fig, use_container_width=True, key=f"batch_chart_{idx}")
                                
                                # 提供下载链接
                                st.markdown("**图表下载选项：**")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    html_link = ExportTool.get_chart_html_download_link(
                                        fig,
                                        f"{class_name}_{student_name}_成绩趋势图.html",
                                        "📥 下载HTML图表"
                                    )
                                    st.markdown(html_link, unsafe_allow_html=True)
                                
                                with col2:
                                    if student_key in student_grades_dict:
                                        student_grades_df = student_grades_dict[student_key]
                                        csv_link = ExportTool.get_chart_data_download_link(
                                            student_grades_df,
                                            f"{class_name}_{student_name}_成绩数据.csv",
                                            "📥 下载数据CSV"
                                        )
                                        st.markdown(csv_link, unsafe_allow_html=True)
                                
                                st.markdown("---")
                    
                    # 批量下载功能
                    st.markdown("#### 💾 批量查询结果导出")
                    
                    batch_excel_data = ExportTool.convert_batch_to_excel(batch_results)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if batch_excel_data:
                            st.download_button(
                                label="📥 下载合并Excel",
                                data=batch_excel_data,
                                file_name=f"批量查询_成绩表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="batch_excel_download"
                            )
                    
                    with col2:
                        batch_csv_data = batch_results.to_csv(index=False, encoding='utf-8-sig')
                        
                        st.download_button(
                            label="📥 下载合并CSV",
                            data=batch_csv_data,
                            file_name=f"批量查询_成绩表_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="batch_csv_download"
                        )
                    
                    with col3:
                        # 批量下载所有图表为HTML
                        if st.session_state.batch_student_charts:
                            zip_html_data = ExportTool.create_charts_zip_html(st.session_state.batch_student_charts)
                            if zip_html_data:
                                st.download_button(
                                    label="📦 下载所有HTML图表",
                                    data=zip_html_data,
                                    file_name=f"批量查询_成绩趋势图_HTML_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                    mime="application/zip",
                                    key="batch_html_zip_download"
                                )
                    
                    with col4:
                        # 批量下载所有图表为PDF
                        if student_grades_dict and st.session_state.batch_global_subjects:
                            # 创建PDF
                            pdf_data = PDFGenerator.create_pdf_with_charts(
                                student_grades_dict,
                                st.session_state.batch_global_subjects,
                                st.session_state.charts_per_page_value
                            )
                            
                            if pdf_data:
                                st.download_button(
                                    label="📄 下载合并PDF",
                                    data=pdf_data,
                                    file_name=f"批量查询_成绩趋势图_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    key="batch_pdf_download"
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
        
        batch_display_df = format_grades_dataframe(st.session_state.batch_results)
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
                    st.plotly_chart(fig, use_container_width=True, key=f"history_batch_chart_{idx}")
                    st.markdown("---")

# ============================================
# 模块4: 学生成绩分析、预测（智能版）
# 结合趋势感知、迁移学习和聚类分析的智能预测系统
# ============================================



# ============================================
# 智能成绩预测器类
# 结合趋势感知、迁移学习和聚类分析
# ============================================
class IntelligentGradePredictor:
    """
    智能成绩预测器
    结合多种技术解决小样本过拟合和趋势识别问题
    """
    
    def __init__(self, all_students_data: pd.DataFrame = None):
        """
        初始化智能预测器
        
        Args:
            all_students_data: 所有学生数据（用于迁移学习和聚类）
        """
        self.all_students_data = all_students_data
        self.pattern_cache = {}  # 模式缓存
        self.cluster_models = {}  # 聚类模型缓存
        
    def analyze_student_pattern(self, student_scores: np.ndarray) -> Dict[str, Any]:
        """
        深度分析学生成绩模式
        
        Args:
            student_scores: 学生成绩数组
            
        Returns:
            模式分析字典
        """
        if len(student_scores) < 2:
            return {"trend": "insufficient_data", "pattern": "unknown"}
        
        # 1. 计算基本统计
        scores = student_scores.astype(float)
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0
        
        # 2. 计算短期和长期趋势
        if len(scores) >= 2:
            # 整体趋势
            time_indices = np.arange(len(scores))
            slope, intercept = np.polyfit(time_indices, scores, 1)
            
            # 近期趋势（最近3次考试）
            recent_slope = 0
            if len(scores) >= 3:
                recent_indices = np.arange(max(0, len(scores)-3), len(scores))
                recent_scores = scores[-3:]
                if len(recent_scores) >= 2:
                    recent_slope, _ = np.polyfit(recent_indices - recent_indices[0], recent_scores, 1)
            
            # 趋势强度
            trend_strength = abs(slope) / (std_score + 1e-6) if std_score > 0 else 0
            
            # 3. 检测模式变化点
            has_pattern_change = False
            change_point = None
            
            if len(scores) >= 4:
                # 使用滑动窗口检测变化
                window_size = min(3, len(scores) // 2)
                for i in range(window_size, len(scores) - window_size + 1):
                    before_mean = np.mean(scores[:i])
                    after_mean = np.mean(scores[i:])
                    if abs(after_mean - before_mean) > 2 * std_score and std_score > 0:
                        has_pattern_change = True
                        change_point = i
                        break
            
            # 4. 确定模式类型
            pattern_type = "unknown"
            
            if trend_strength > 0.3:  # 强趋势
                if slope > 0.5:
                    pattern_type = "strong_upward"
                elif slope < -0.5:
                    pattern_type = "strong_downward"
                elif abs(slope) <= 0.5:
                    pattern_type = "stable"
            else:  # 弱趋势
                if abs(recent_slope) > 0.5:  # 近期有明显趋势
                    if recent_slope > 0:
                        pattern_type = "recent_upward"
                    else:
                        pattern_type = "recent_downward"
                else:
                    if std_score < 5:  # 低波动
                        pattern_type = "stable_low_volatility"
                    else:  # 高波动
                        pattern_type = "volatile"
            
            # 考虑模式变化
            if has_pattern_change and change_point:
                if pattern_type in ["stable", "stable_low_volatility"]:
                    pattern_type = f"changed_at_{change_point}"
            
            return {
                "pattern_type": pattern_type,
                "slope": float(slope),
                "recent_slope": float(recent_slope),
                "mean": float(mean_score),
                "std": float(std_score),
                "trend_strength": float(trend_strength),
                "has_pattern_change": has_pattern_change,
                "change_point": change_point,
                "data_points": len(scores)
            }
        
        return {"pattern": "insufficient_data"}
    
    def adaptive_weighted_average(self, scores: np.ndarray, pattern_info: Dict) -> np.ndarray:
        """
        自适应加权平均
        根据模式动态调整权重
        
        Args:
            scores: 成绩数组
            pattern_info: 模式信息
            
        Returns:
            权重数组
        """
        n = len(scores)
        
        # 基础权重：给近期更高权重
        base_weights = np.linspace(0.5, 1.5, n)
        
        # 根据模式调整权重
        pattern_type = pattern_info.get("pattern_type", "unknown")
        
        if pattern_type in ["strong_upward", "recent_upward"]:
            # 上升趋势：给近期成绩更高权重
            adjusted_weights = base_weights * np.exp(np.linspace(-0.5, 0.5, n))
        elif pattern_type in ["strong_downward", "recent_downward"]:
            # 下降趋势：适当降低近期权重，防止过度悲观
            if pattern_info.get("trend_strength", 0) > 0.5:
                # 强下降趋势，降低近期权重
                adjusted_weights = base_weights * np.exp(np.linspace(0.3, -0.3, n))
            else:
                # 弱下降趋势，保持权重
                adjusted_weights = base_weights
        elif pattern_type in ["stable", "stable_low_volatility"]:
            # 稳定模式：更均匀的权重
            adjusted_weights = np.ones(n)
        elif "changed_at" in pattern_type:
            # 模式变化：变化点后的数据权重更高
            change_point = pattern_info.get("change_point", n//2)
            adjusted_weights = np.ones(n)
            adjusted_weights[change_point:] *= 1.5
        else:
            # 波动模式：降低极端值的权重
            adjusted_weights = base_weights * 0.8 + 0.2
        
        # 归一化权重
        if adjusted_weights.sum() > 0:
            adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        return adjusted_weights
    
    def find_similar_students(self, target_scores: np.ndarray, all_students_scores: Dict[str, np.ndarray], 
                             n_similar: int = 5) -> List[Tuple[str, float]]:
        """
        寻找相似学生（迁移学习）
        
        Args:
            target_scores: 目标学生成绩
            all_students_scores: 所有学生成绩字典 {学生名: 成绩数组}
            n_similar: 返回的最相似学生数量
            
        Returns:
            相似学生列表 [(学生名, 相似度分数), ...]
        """
        if len(all_students_scores) == 0 or len(target_scores) < 2:
            return []
        
        similarities = []
        target_len = len(target_scores)
        
        for student_name, scores in all_students_scores.items():
            if len(scores) < 2:
                continue
            
            # 截取相同长度的成绩序列进行比较
            compare_len = min(target_len, len(scores))
            target_subset = target_scores[-compare_len:]
            student_subset = scores[-compare_len:]
            
            # 计算相似度（使用多种指标的综合）
            similarity_score = self.calculate_similarity(target_subset, student_subset)
            
            if similarity_score > 0.3:  # 相似度阈值
                similarities.append((student_name, similarity_score))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_similar]
    
    def calculate_similarity(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """
        计算两个成绩序列的相似度
        
        Args:
            scores1: 成绩序列1
            scores2: 成绩序列2
            
        Returns:
            相似度分数 (0-1)
        """
        if len(scores1) != len(scores2) or len(scores1) < 2:
            return 0.0
        
        # 1. 皮尔逊相关系数
        try:
            correlation, _ = stats.pearsonr(scores1, scores2)
            correlation_score = max(0, correlation)  # 负相关视为不相似
        except:
            correlation_score = 0.0
        
        # 2. 趋势相似度
        slope1, _ = np.polyfit(np.arange(len(scores1)), scores1, 1)
        slope2, _ = np.polyfit(np.arange(len(scores2)), scores2, 1)
        trend_similarity = 1.0 - min(1.0, abs(slope1 - slope2) / 2.0)
        
        # 3. 水平相似度
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        level_similarity = 1.0 - min(1.0, abs(mean1 - mean2) / 20.0)
        
        # 4. 波动性相似度
        std1, std2 = np.std(scores1) if len(scores1) > 1 else 0, np.std(scores2) if len(scores2) > 1 else 0
        volatility_similarity = 1.0 - min(1.0, abs(std1 - std2) / 10.0)
        
        # 综合相似度
        weights = [0.4, 0.3, 0.2, 0.1]  # 相关性权重最高
        similarity = (weights[0] * correlation_score + 
                     weights[1] * trend_similarity + 
                     weights[2] * level_similarity + 
                     weights[3] * volatility_similarity)
        
        return similarity
    
    def cluster_students_by_pattern(self, all_students_scores: Dict[str, np.ndarray], 
                                  n_clusters: int = 3) -> Dict[str, int]:
        """
        按成绩模式对学生进行聚类
        
        Args:
            all_students_scores: 所有学生成绩字典
            n_clusters: 聚类数量
            
        Returns:
            聚类结果字典 {学生名: 聚类标签}
        """
        if len(all_students_scores) < n_clusters * 2:  # 样本太少不适合聚类
            return {name: 0 for name in all_students_scores.keys()}
        
        # 提取特征
        features = []
        student_names = []
        
        for name, scores in all_students_scores.items():
            if len(scores) >= 3:  # 至少3个数据点
                # 特征：均值、标准差、斜率、最近变化
                mean = np.mean(scores)
                std = np.std(scores) if len(scores) > 1 else 0
                
                # 计算斜率
                if len(scores) >= 2:
                    time_indices = np.arange(len(scores))
                    slope, _ = np.polyfit(time_indices, scores, 1)
                else:
                    slope = 0
                
                # 最近变化
                if len(scores) >= 2:
                    recent_change = scores[-1] - scores[-2]
                else:
                    recent_change = 0
                
                features.append([mean, std, slope, recent_change])
                student_names.append(name)
        
        if len(features) < n_clusters:
            return {name: 0 for name in all_students_scores.keys()}
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 聚类
        try:
            kmeans = KMeans(n_clusters=min(n_clusters, len(features)), 
                          random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # 计算轮廓系数
            if len(set(cluster_labels)) > 1:
                silhouette_avg = silhouette_score(features_scaled, cluster_labels)
            else:
                silhouette_avg = 0.0
            
            # 创建聚类结果字典
            clustering_result = {}
            for i, name in enumerate(student_names):
                clustering_result[name] = int(cluster_labels[i])
            
            # 为没有足够数据的学生分配默认聚类
            for name in all_students_scores.keys():
                if name not in clustering_result:
                    clustering_result[name] = 0
            
            # 缓存聚类信息
            self.cluster_models['kmeans'] = kmeans
            self.cluster_models['scaler'] = scaler
            self.cluster_models['silhouette'] = silhouette_avg
            
            return clustering_result
            
        except Exception as e:
            # 聚类失败，返回默认聚类
            return {name: 0 for name in all_students_scores.keys()}
    
    def predict_with_transfer_learning(self, target_scores: np.ndarray, 
                                      similar_students: List[Tuple[str, float]],
                                      all_students_scores: Dict[str, np.ndarray]) -> float:
        """
        使用迁移学习进行预测
        
        Args:
            target_scores: 目标学生成绩
            similar_students: 相似学生列表
            all_students_scores: 所有学生成绩字典
            
        Returns:
            预测成绩
        """
        if not similar_students:
            return None
        
        predictions = []
        weights = []
        
        for student_name, similarity in similar_students:
            if similarity < 0.4:  # 相似度阈值
                continue
                
            student_scores = all_students_scores[student_name]
            if len(student_scores) <= len(target_scores):
                # 相似学生的后续成绩不可用
                continue
            
            # 使用相似学生的后续成绩作为参考
            next_score = student_scores[len(target_scores)]  # 对应位置的下一次成绩
            predictions.append(next_score)
            weights.append(similarity)
        
        if not predictions:
            return None
        
        # 加权平均
        weights = np.array(weights)
        predictions = np.array(predictions)
        
        if weights.sum() > 0:
            weighted_prediction = np.sum(predictions * weights) / weights.sum()
            return float(weighted_prediction)
        
        return None
    
    def trend_aware_prediction(self, scores: np.ndarray, pattern_info: Dict) -> Dict[str, Any]:
        """
        趋势感知预测
        
        Args:
            scores: 成绩数组
            pattern_info: 模式信息
            
        Returns:
            预测结果
        """
        n = len(scores)
        
        # 根据模式选择预测方法
        pattern_type = pattern_info.get("pattern_type", "unknown")
        slope = pattern_info.get("slope", 0)
        recent_slope = pattern_info.get("recent_slope", 0)
        trend_strength = pattern_info.get("trend_strength", 0)
        
        # 基础预测（加权平均）
        weights = self.adaptive_weighted_average(scores, pattern_info)
        base_prediction = np.sum(scores * weights)
        
        # 趋势调整
        trend_adjustment = 0
        
        if pattern_type in ["strong_upward", "recent_upward"]:
            # 上升趋势：基于斜率调整
            if abs(slope) > 0.2:
                trend_adjustment = slope * 0.8  # 调整幅度
        elif pattern_type in ["strong_downward", "recent_downward"]:
            # 下降趋势：谨慎调整
            if abs(slope) > 0.3:
                trend_adjustment = slope * 0.5
        elif pattern_type == "stable":
            # 稳定模式：小幅调整
            trend_adjustment = recent_slope * 0.3
        elif "changed_at" in pattern_type:
            # 模式变化：主要基于近期数据
            change_point = pattern_info.get("change_point", n//2)
            recent_scores = scores[change_point:] if change_point < n else scores
            if len(recent_scores) >= 2:
                recent_slope, _ = np.polyfit(np.arange(len(recent_scores)), recent_scores, 1)
                trend_adjustment = recent_slope * 0.7
        
        # 最终预测
        final_prediction = base_prediction + trend_adjustment
        
        # 置信度计算
        data_points = n
        volatility = pattern_info.get("std", 5)
        
        data_sufficiency = min(1.0, data_points / 6.0)
        stability = max(0.1, 1.0 - volatility / 15.0)
        
        # 模式置信度
        if pattern_type in ["strong_upward", "strong_downward", "stable"]:
            pattern_confidence = 0.8
        elif pattern_type in ["recent_upward", "recent_downward"]:
            pattern_confidence = 0.6
        else:
            pattern_confidence = 0.4
        
        confidence = 0.3 + 0.3 * data_sufficiency + 0.2 * stability + 0.2 * pattern_confidence
        confidence = min(0.9, max(0.3, confidence))
        
        confidence_interval = 2 + 1.5 * volatility
        
        return {
            'prediction': float(final_prediction),
            'confidence_interval': float(confidence_interval),
            'confidence_score': float(confidence),
            'pattern': pattern_type,
            'slope': float(slope),
            'trend_strength': float(trend_strength),
            'method': '趋势感知预测',
            'base_prediction': float(base_prediction),
            'trend_adjustment': float(trend_adjustment)
        }
    
    def smart_predict(self, scores: np.ndarray, student_name: str = None, 
                     all_students_data: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """
        智能预测主函数
        
        Args:
            scores: 成绩数组
            student_name: 学生姓名（用于迁移学习）
            all_students_data: 所有学生数据
            
        Returns:
            预测结果字典
        """
        if len(scores) < 2:
            return {
                'prediction': float(np.mean(scores)) if len(scores) > 0 else 0,
                'confidence_interval': 10.0,
                'confidence_score': 0.3,
                'method': '数据不足',
                'pattern': 'insufficient_data',
                'data_points': len(scores)
            }
        
        # 1. 分析学生模式
        pattern_info = self.analyze_student_pattern(scores)
        
        # 2. 趋势感知预测
        trend_result = self.trend_aware_prediction(scores, pattern_info)
        
        # 3. 如果数据足够，尝试迁移学习
        transfer_learning_prediction = None
        if student_name and all_students_data and len(all_students_data) >= 5:
            # 寻找相似学生
            similar_students = self.find_similar_students(
                scores, all_students_data, n_similar=3
            )
            
            if similar_students:
                # 迁移学习预测
                transfer_learning_prediction = self.predict_with_transfer_learning(
                    scores, similar_students, all_students_data
                )
                
                if transfer_learning_prediction is not None:
                    # 计算迁移学习的置信度
                    similarity_scores = [sim for _, sim in similar_students]
                    avg_similarity = np.mean(similarity_scores)
                    
                    # 融合迁移学习和趋势感知预测
                    transfer_weight = min(0.5, avg_similarity * 0.7)  # 迁移学习权重
                    trend_weight = 1.0 - transfer_weight
                    
                    # 融合预测
                    final_prediction = (
                        transfer_weight * transfer_learning_prediction +
                        trend_weight * trend_result['prediction']
                    )
                    
                    # 更新结果
                    trend_result['prediction'] = float(final_prediction)
                    trend_result['method'] = '融合预测(趋势+迁移)'
                    trend_result['transfer_learning'] = True
                    trend_result['similar_students_count'] = len(similar_students)
                    trend_result['avg_similarity'] = float(avg_similarity)
                    trend_result['transfer_prediction'] = float(transfer_learning_prediction)
        
        # 4. 边界检查和合理性约束
        final_prediction = trend_result['prediction']
        min_score, max_score = np.min(scores), np.max(scores)
        
        # 确保预测在合理范围内
        if pattern_info.get("pattern_type") in ["strong_upward", "recent_upward"]:
            # 上升趋势：预测不低于近期平均
            recent_avg = np.mean(scores[-min(3, len(scores)):])
            final_prediction = max(final_prediction, recent_avg * 0.95)
        elif pattern_info.get("pattern_type") in ["strong_downward", "recent_downward"]:
            # 下降趋势：预测不高于近期平均
            recent_avg = np.mean(scores[-min(3, len(scores)):])
            final_prediction = min(final_prediction, recent_avg * 1.05)
        
        # 最终边界检查
        score_range = max_score - min_score
        lower_bound = min_score - 0.3 * score_range
        upper_bound = max_score + 0.3 * score_range
        final_prediction = max(lower_bound, min(upper_bound, final_prediction))
        
        trend_result['prediction'] = float(final_prediction)
        trend_result['pattern_info'] = pattern_info
        trend_result['data_points'] = len(scores)
        
        return trend_result


# ============================================
# 班级智能分析器
# ============================================
class ClassIntelligenceAnalyzer:
    """班级智能分析器，支持聚类和模式分析"""
    
    def __init__(self):
        self.patterns_cache = {}
        self.clusters_cache = {}
        
    def analyze_class_patterns(self, class_data: pd.DataFrame, class_name: str, 
                              subject: str) -> Dict[str, Any]:
        """
        分析班级整体的成绩模式
        
        Args:
            class_data: 班级数据
            class_name: 班级名称
            subject: 科目
            
        Returns:
            班级模式分析结果
        """
        # 提取所有学生成绩
        student_scores = {}
        student_names = class_data[st.session_state.name_column_name].unique()
        
        for student in student_names:
            grades_df = GradeManager.get_student_grades(
                class_data, class_name, student,
                st.session_state.class_column_name, st.session_state.name_column_name,
                st.session_state.subjects, st.session_state.exams,
                st.session_state.column_mapping
            )
            
            if grades_df is not None and subject in grades_df.columns:
                scores = pd.to_numeric(grades_df[subject], errors='coerce').dropna().values
                if len(scores) >= 2:
                    student_scores[student] = scores
        
        if len(student_scores) < 3:
            return {"error": "数据不足"}
        
        # 使用智能预测器分析模式
        predictor = IntelligentGradePredictor()
        pattern_distribution = defaultdict(int)
        
        for student, scores in student_scores.items():
            pattern_info = predictor.analyze_student_pattern(scores)
            pattern_type = pattern_info.get("pattern_type", "unknown")
            pattern_distribution[pattern_type] += 1
        
        # 聚类分析
        clustering = predictor.cluster_students_by_pattern(student_scores, n_clusters=3)
        
        # 分析每个聚类的特征
        cluster_features = {}
        for student, cluster_label in clustering.items():
            if student in student_scores:
                scores = student_scores[student]
                if cluster_label not in cluster_features:
                    cluster_features[cluster_label] = {
                        'students': [],
                        'avg_scores': [],
                        'patterns': []
                    }
                
                pattern_info = predictor.analyze_student_pattern(scores)
                
                cluster_features[cluster_label]['students'].append(student)
                cluster_features[cluster_label]['avg_scores'].append(np.mean(scores))
                cluster_features[cluster_label]['patterns'].append(pattern_info.get("pattern_type", "unknown"))
        
        # 计算每个聚类的统计
        for cluster_label, features in cluster_features.items():
            if features['avg_scores']:
                features['avg_score'] = np.mean(features['avg_scores'])
                features['size'] = len(features['students'])
                
                # 主要模式
                pattern_counts = {}
                for pattern in features['patterns']:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                if pattern_counts:
                    main_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]
                    features['main_pattern'] = main_pattern
        
        return {
            'student_count': len(student_scores),
            'pattern_distribution': dict(pattern_distribution),
            'clustering': clustering,
            'cluster_features': cluster_features,
            'student_scores': student_scores  # 用于预测
        }
    
    def get_similar_students_for_prediction(self, class_patterns: Dict, 
                                           target_student: str, n_similar: int = 5) -> List[Tuple[str, float]]:
        """
        为目标学生获取相似学生
        
        Args:
            class_patterns: 班级模式分析结果
            target_student: 目标学生
            n_similar: 返回的相似学生数量
            
        Returns:
            相似学生列表
        """
        if 'student_scores' not in class_patterns or target_student not in class_patterns['student_scores']:
            return []
        
        target_scores = class_patterns['student_scores'][target_student]
        all_scores = class_patterns['student_scores']
        
        # 移除目标学生
        other_scores = {k: v for k, v in all_scores.items() if k != target_student}
        
        if not other_scores:
            return []
        
        # 使用智能预测器寻找相似学生
        predictor = IntelligentGradePredictor()
        similar_students = predictor.find_similar_students(
            target_scores, other_scores, n_similar
        )
        
        return similar_students


# ============================================
# 智能批量预测（优化版）
# ============================================
def intelligent_batch_prediction():
    """
    智能批量预测 - 结合聚类和迁移学习
    """
    st.markdown("#### 🧠 智能批量成绩预测")
    st.markdown("结合趋势感知、聚类分析和迁移学习的智能预测系统")
    
    # 选择班级
    classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
    
    if not classes:
        st.warning("没有找到班级数据")
        return
    
    selected_class = st.selectbox(
        "选择班级",
        classes,
        key="intelligent_batch_class_select"
    )
    
    if not selected_class:
        return
    
    # 获取班级学生列表
    class_data = st.session_state.df[
        st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == selected_class
    ]
    class_students = class_data[st.session_state.name_column_name].dropna().unique()
    
    if len(class_students) == 0:
        st.warning("该班级没有学生数据")
        return
    
    # 选择要预测的科目
    sample_student = class_students[0]
    sample_grades_df = GradeManager.get_student_grades(
        class_data, selected_class, sample_student,
        st.session_state.class_column_name, st.session_state.name_column_name,
        st.session_state.subjects, st.session_state.exams,
        st.session_state.column_mapping
    )
    
    if sample_grades_df is None or sample_grades_df.empty:
        st.error("无法获取学生成绩数据")
        return
    
    score_subjects = [s for s in st.session_state.subjects if '排' not in s and s in sample_grades_df.columns]
    
    if not score_subjects:
        st.info("没有找到可预测的成绩科目")
        return
    
    selected_subject = st.selectbox(
        "选择要预测的科目",
        score_subjects,
        key="intelligent_batch_subject"
    )
    
    # 选择预测模式
    st.markdown("#### 🎯 预测模式设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_transfer_learning = st.checkbox(
            "使用迁移学习", 
            value=True,
            help="使用相似学生的成绩模式进行预测，可提高准确性"
        )
        
        use_clustering = st.checkbox(
            "使用聚类分析", 
            value=True,
            help="将学生按成绩模式聚类，提供更精准的预测"
        )
    
    with col2:
        min_data_points = st.slider(
            "最小数据点要求", 
            min_value=2, max_value=6, value=3,
            help="进行预测所需的最少考试次数"
        )
        
        confidence_threshold = st.slider(
            "置信度阈值", 
            min_value=0.3, max_value=0.9, value=0.6, step=0.1,
            help="只显示置信度高于此值的预测"
        )
    
    # 选择学生
    st.markdown("#### 👥 选择学生")
    prediction_mode = st.radio(
        "预测范围",
        ["选择学生", "全班预测", "按聚类选择"],
        key="intelligent_prediction_mode"
    )
    
    selected_students = []
    
    if prediction_mode == "选择学生":
        selected_students = st.multiselect(
            "选择要预测的学生（可多选）",
            sorted(class_students),
            default=sorted(class_students)[:min(5, len(class_students))],
            key="intelligent_selected_students"
        )
        
    elif prediction_mode == "全班预测":
        selected_students = sorted(class_students)
        st.info(f"将预测全班 {len(selected_students)} 名学生")
        
    elif prediction_mode == "按聚类选择":
        # 先进行聚类分析
        with st.spinner("正在进行班级聚类分析..."):
            analyzer = ClassIntelligenceAnalyzer()
            class_patterns = analyzer.analyze_class_patterns(
                class_data, selected_class, selected_subject
            )
        
        if 'error' in class_patterns:
            st.warning(f"聚类分析失败: {class_patterns['error']}")
            selected_students = sorted(class_students)
        else:
            # 显示聚类结果
            st.markdown("##### 📊 班级聚类分析结果")
            
            cluster_features = class_patterns.get('cluster_features', {})
            
            cols = st.columns(min(3, len(cluster_features)))
            
            for idx, (cluster_id, features) in enumerate(cluster_features.items()):
                with cols[idx % len(cols)]:
                    with st.expander(f"聚类 {cluster_id+1} ({features.get('size', 0)}人)"):
                        st.metric("平均成绩", f"{features.get('avg_score', 0):.1f}")
                        st.metric("主要模式", features.get('main_pattern', '未知'))
                        st.metric("学生数", features.get('size', 0))
            
            # 选择聚类
            available_clusters = list(cluster_features.keys())
            selected_cluster = st.selectbox(
                "选择要预测的聚类",
                available_clusters,
                format_func=lambda x: f"聚类 {x+1} ({cluster_features[x].get('size', 0)}人)",
                key="cluster_selection"
            )
            
            if selected_cluster in cluster_features:
                cluster_students = cluster_features[selected_cluster].get('students', [])
                selected_students = st.multiselect(
                    f"选择聚类 {selected_cluster+1} 中的学生",
                    sorted(cluster_students),
                    default=sorted(cluster_students)[:min(10, len(cluster_students))],
                    key="cluster_students_selection"
                )
                
                # 缓存聚类结果用于预测
                st.session_state['class_patterns'] = class_patterns
    
    if not selected_students:
        st.warning("请选择至少一名学生")
        return
    
    # 开始预测按钮
    if st.button("🚀 开始智能预测", type="primary", key="start_intelligent_prediction"):
        if not selected_subject:
            st.warning("请选择要预测的科目")
            return
        
        st.markdown(f"### 🧠 智能预测结果 - {selected_class} - {selected_subject}")
        
        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 收集所有学生成绩数据（用于迁移学习）
        all_students_scores = {}
        student_patterns = {}
        
        with st.spinner("收集班级成绩数据..."):
            for student in class_students:
                grades_df = GradeManager.get_student_grades(
                    class_data, selected_class, student,
                    st.session_state.class_column_name, st.session_state.name_column_name,
                    st.session_state.subjects, st.session_state.exams,
                    st.session_state.column_mapping
                )
                
                if grades_df is not None and selected_subject in grades_df.columns:
                    scores = pd.to_numeric(grades_df[selected_subject], errors='coerce').dropna().values
                    if len(scores) >= 2:
                        all_students_scores[student] = scores
        
        # 如果需要，进行聚类分析
        if use_clustering and 'class_patterns' in st.session_state:
            class_patterns = st.session_state['class_patterns']
        elif use_clustering:
            with st.spinner("进行聚类分析..."):
                analyzer = ClassIntelligenceAnalyzer()
                class_patterns = analyzer.analyze_class_patterns(
                    class_data, selected_class, selected_subject
                )
            st.session_state['class_patterns'] = class_patterns
        else:
            class_patterns = {}
        
        # 进行预测
        prediction_results = []
        valid_predictions = 0
        high_confidence_predictions = 0
        
        for idx, student_name in enumerate(selected_students):
            # 更新进度
            progress = (idx + 1) / len(selected_students)
            progress_bar.progress(progress)
            status_text.text(f"智能预测中: {student_name} ({idx+1}/{len(selected_students)})")
            
            try:
                # 获取学生成绩
                grades_df = GradeManager.get_student_grades(
                    class_data, selected_class, student_name,
                    st.session_state.class_column_name, st.session_state.name_column_name,
                    st.session_state.subjects, st.session_state.exams,
                    st.session_state.column_mapping
                )
                
                if grades_df is not None and selected_subject in grades_df.columns:
                    scores = pd.to_numeric(grades_df[selected_subject], errors='coerce').dropna().values
                    
                    if len(scores) >= min_data_points:
                        # 使用智能预测器
                        predictor = IntelligentGradePredictor(all_students_scores)
                        
                        # 准备迁移学习数据
                        transfer_data = all_students_scores if use_transfer_learning else None
                        
                        # 进行预测
                        prediction_result = predictor.smart_predict(
                            scores, student_name, transfer_data
                        )
                        
                        if prediction_result and 'prediction' in prediction_result:
                            # 获取当前成绩
                            current_grade = scores[-1] if len(scores) > 0 else 0
                            prediction_value = prediction_result['prediction']
                            confidence_score = prediction_result.get('confidence_score', 0)
                            pattern_type = prediction_result.get('pattern', 'unknown')
                            
                            # 计算变化
                            improvement = prediction_value - current_grade
                            
                            # 确定趋势
                            if improvement > 1.0:
                                trend = "上升"
                                trend_level = "显著上升" if improvement > 5.0 else "轻微上升"
                            elif improvement < -1.0:
                                trend = "下降"
                                trend_level = "显著下降" if improvement < -5.0 else "轻微下降"
                            else:
                                trend = "稳定"
                                trend_level = "基本持平"
                            
                            # 模式描述
                            pattern_map = {
                                "strong_upward": "强上升趋势",
                                "recent_upward": "近期上升",
                                "strong_downward": "强下降趋势",
                                "recent_downward": "近期下降",
                                "stable": "稳定",
                                "stable_low_volatility": "低波动稳定",
                                "volatile": "高波动",
                                "changed_at": "模式转变"
                            }
                            pattern_desc = pattern_map.get(pattern_type, pattern_type)
                            
                            # 置信度级别
                            if confidence_score >= 0.7:
                                confidence_level = "高"
                            elif confidence_score >= 0.5:
                                confidence_level = "中"
                            else:
                                confidence_level = "低"
                            
                            result = {
                                '序号': idx + 1,
                                '姓名': student_name,
                                '当前成绩': round(current_grade, 1),
                                '预测成绩': round(prediction_value, 1),
                                '预测变化': round(improvement, 1),
                                '变化趋势': trend_level,
                                '成绩模式': pattern_desc,
                                '预测方法': prediction_result.get('method', '智能预测'),
                                '置信度': f"{confidence_score:.1%}",
                                '置信度级别': confidence_level,
                                '有效考试次数': len(scores),
                                '数据充足度': "充足" if len(scores) >= 5 else "中等" if len(scores) >= 3 else "较少"
                            }
                            
                            # 添加迁移学习信息
                            if prediction_result.get('transfer_learning', False):
                                result['预测方法'] = '融合预测'
                                result['相似学生数'] = prediction_result.get('similar_students_count', 0)
                                result['平均相似度'] = f"{prediction_result.get('avg_similarity', 0):.1%}"
                            
                            prediction_results.append(result)
                            
                            if confidence_score >= confidence_threshold:
                                high_confidence_predictions += 1
                            valid_predictions += 1
                            
                        else:
                            prediction_results.append({
                                '序号': idx + 1,
                                '姓名': student_name,
                                '当前成绩': round(scores[-1], 1) if len(scores) > 0 else "N/A",
                                '预测成绩': "N/A",
                                '预测变化': "N/A",
                                '变化趋势': "预测失败",
                                '成绩模式': "N/A",
                                '预测方法': "N/A",
                                '置信度': "N/A",
                                '置信度级别': "N/A",
                                '有效考试次数': len(scores),
                                '数据充足度': "不足" if len(scores) < min_data_points else "较少"
                            })
                    else:
                        prediction_results.append({
                            '序号': idx + 1,
                            '姓名': student_name,
                            '当前成绩': round(scores[-1], 1) if len(scores) > 0 else "N/A",
                            '预测成绩': "N/A",
                            '预测变化': "N/A",
                            '变化趋势': "数据不足",
                            '成绩模式': "N/A",
                            '预测方法': "N/A",
                            '置信度': "N/A",
                            '置信度级别': "N/A",
                            '有效考试次数': len(scores),
                            '数据充足度': "不足"
                        })
                else:
                    prediction_results.append({
                        '序号': idx + 1,
                        '姓名': student_name,
                        '当前成绩': "N/A",
                        '预测成绩': "N/A",
                        '预测变化': "N/A",
                        '变化趋势': "无成绩数据",
                        '成绩模式': "N/A",
                        '预测方法': "N/A",
                        '置信度': "N/A",
                        '置信度级别': "N/A",
                        '有效考试次数': 0,
                        '数据充足度': "无数据"
                    })
                    
            except Exception as e:
                prediction_results.append({
                    '序号': idx + 1,
                    '姓名': student_name,
                    '当前成绩': "N/A",
                    '预测成绩': "N/A",
                    '预测变化': "N/A",
                    '变化趋势': f"错误: {str(e)[:30]}...",
                    '成绩模式': "N/A",
                    '预测方法': "N/A",
                    '置信度': "N/A",
                    '置信度级别': "N/A",
                    '有效考试次数': 0,
                    '数据充足度': "错误"
                })
        
        # 完成进度
        progress_bar.empty()
        status_text.empty()
        
        # 显示预测结果
        display_intelligent_results(
            prediction_results, valid_predictions, high_confidence_predictions,
            len(selected_students), selected_class, selected_subject,
            confidence_threshold
        )


def display_intelligent_results(results, valid_count, high_confidence_count, total_count,
                               class_name, subject, confidence_threshold):
    """显示智能预测结果"""
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 显示统计
    st.markdown("##### 📊 智能预测统计")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("预测学生总数", total_count)
    with col2:
        st.metric("成功预测数", valid_count)
    with col3:
        success_rate = (valid_count / total_count * 100) if total_count > 0 else 0
        st.metric("预测成功率", f"{success_rate:.1f}%")
    with col4:
        high_conf_rate = (high_confidence_count / valid_count * 100) if valid_count > 0 else 0
        st.metric(f"高置信度(≥{confidence_threshold:.0%})", f"{high_conf_rate:.1f}%")
    
    # 显示详细结果
    st.markdown("##### 📋 详细预测结果")
    
    # 添加筛选选项
    col1, col2 = st.columns(2)
    with col1:
        show_all = st.checkbox("显示所有预测", value=True, key="show_all_predictions")
    
    with col2:
        if not show_all:
            min_confidence = st.slider("最小置信度", 0.0, 1.0, 0.6, 0.05, key="min_confidence_filter")
    
    # 筛选结果
    if show_all:
        filtered_results = results_df
    else:
        # 过滤低置信度预测
        def get_confidence(conf_str):
            try:
                return float(conf_str.strip('%')) / 100
            except:
                return 0.0
        
        filtered_results = results_df[
            results_df['置信度'].apply(
                lambda x: get_confidence(x) >= min_confidence if isinstance(x, str) and x != 'N/A' else False
            )
        ]
    
    st.dataframe(filtered_results, use_container_width=True, height=400)
    
    # 模式分析
    if valid_count > 0:
        st.markdown("##### 📈 成绩模式分析")
        
        # 提取有效结果
        valid_results = [r for r in results if isinstance(r.get('预测成绩'), (int, float))]
        
        if valid_results:
            # 模式分布
            pattern_counts = {}
            for result in valid_results:
                pattern = result.get('成绩模式', '未知')
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
            # 趋势分布
            trend_counts = {}
            for result in valid_results:
                trend = result.get('变化趋势', '未知')
                trend_counts[trend] = trend_counts.get(trend, 0) + 1
            
            col1, col2 = st.columns(2)
            
            with col1:
                if pattern_counts:
                    fig_pattern = go.Figure(data=[go.Pie(
                        labels=list(pattern_counts.keys()),
                        values=list(pattern_counts.values()),
                        hole=.3,
                        textinfo='label+percent',
                        marker=dict(colors=px.colors.qualitative.Set3)
                    )])
                    fig_pattern.update_layout(
                        title="学生成绩模式分布",
                        height=300,
                        showlegend=True
                    )
                    st.plotly_chart(fig_pattern, use_container_width=True)
            
            with col2:
                if trend_counts:
                    fig_trend = go.Figure(data=[go.Pie(
                        labels=list(trend_counts.keys()),
                        values=list(trend_counts.values()),
                        hole=.3,
                        textinfo='label+percent',
                        marker=dict(colors=['green', 'orange', 'red', 'blue'])
                    )])
                    fig_trend.update_layout(
                        title="预测变化趋势分布",
                        height=300,
                        showlegend=True
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            # 预测成绩分布
            st.markdown("##### 📊 预测成绩分布")
            
            prediction_scores = [r['预测成绩'] for r in valid_results]
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=prediction_scores,
                nbinsx=20,
                name='预测成绩分布',
                marker_color='skyblue',
                opacity=0.7
            ))
            
            # 添加统计线
            avg_score = np.mean(prediction_scores)
            fig_dist.add_vline(
                x=avg_score, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"平均: {avg_score:.1f}",
                annotation_position="top right"
            )
            
            fig_dist.update_layout(
                title="预测成绩分布直方图",
                xaxis_title="预测成绩",
                yaxis_title="学生人数",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # 显示统计信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("平均预测成绩", f"{avg_score:.1f}")
            with col2:
                st.metric("最高预测成绩", f"{max(prediction_scores):.1f}")
            with col3:
                st.metric("最低预测成绩", f"{min(prediction_scores):.1f}")
            with col4:
                st.metric("成绩标准差", f"{np.std(prediction_scores):.1f}")
    
    # 数据下载
    st.markdown("##### 💾 下载预测结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 转换为CSV
        csv = results_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下载预测结果(CSV)",
            data=csv,
            file_name=f"{class_name}_{subject}_智能预测_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="intelligent_prediction_csv"
        )
    
    with col2:
        # 生成智能报告
        report_text = generate_intelligent_report(results_df, valid_count, high_confidence_count, 
                                                total_count, class_name, subject, confidence_threshold)
        
        st.download_button(
            label="📝 下载智能报告(TXT)",
            data=report_text,
            file_name=f"{class_name}_{subject}_智能报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key="intelligent_prediction_txt"
        )
    
    # 教学建议
    st.markdown("##### 💡 智能教学建议")
    
    if valid_count == 0:
        st.warning("⚠️ 没有成功预测任何学生，可能因为数据不足")
        st.info("""
        **建议**：
        1. 检查学生成绩数据是否完整
        2. 降低最小数据点要求
        3. 确保数据格式正确
        """)
    else:
        # 分析模式给出建议
        valid_results = [r for r in results if isinstance(r.get('预测成绩'), (int, float))]
        
        if valid_results:
            # 统计不同模式
            strong_upward = sum(1 for r in valid_results if r.get('成绩模式') == '强上升趋势')
            recent_upward = sum(1 for r in valid_results if r.get('成绩模式') == '近期上升')
            strong_downward = sum(1 for r in valid_results if r.get('成绩模式') == '强下降趋势')
            recent_downward = sum(1 for r in valid_results if r.get('成绩模式') == '近期下降')
            
            st.success(f"✅ 成功预测 {valid_count} 名学生")
            
            if strong_downward + recent_downward > 0:
                st.warning(f"⚠️ 发现 {strong_downward + recent_downward} 名学生有下降趋势，需要重点关注")
                
            if strong_upward + recent_upward > 0:
                st.info(f"📈 {strong_upward + recent_upward} 名学生呈现上升趋势，可给予鼓励")
            
            st.info("""
            **智能教学建议**：
            1. 关注下降趋势学生，提供个性化辅导
            2. 对上升趋势学生给予正向强化
            3. 对稳定型学生保持现有教学节奏
            4. 对波动型学生分析原因，提供稳定性训练
            5. 利用聚类分析结果进行分组教学
            """)


def generate_intelligent_report(results_df, valid_count, high_confidence_count, total_count,
                               class_name, subject, confidence_threshold):
    """生成智能预测报告"""
    
    # 计算统计
    success_rate = (valid_count / total_count * 100) if total_count > 0 else 0
    high_conf_rate = (high_confidence_count / valid_count * 100) if valid_count > 0 else 0
    
    # 分析方法统计
    method_counts = {}
    pattern_counts = {}
    trend_counts = {}
    
    for _, row in results_df.iterrows():
        if row['预测方法'] != 'N/A':
            method = row['预测方法']
            method_counts[method] = method_counts.get(method, 0) + 1
        
        if row['成绩模式'] != 'N/A':
            pattern = row['成绩模式']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        if row['变化趋势'] not in ['N/A', '数据不足', '无成绩数据', '预测失败']:
            trend = row['变化趋势']
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
    
    # 生成报告
    report_text = f"""
========== 智能成绩预测报告 ==========

班级：{class_name}
科目：{subject}
报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
预测方法：趋势感知 + 迁移学习 + 聚类分析
置信度阈值：{confidence_threshold:.0%}

【预测统计】
预测学生总数：{total_count}
成功预测数：{valid_count}
预测成功率：{success_rate:.1f}%
高置信度预测数：{high_confidence_count}
高置信度比例：{high_conf_rate:.1f}%

【预测方法统计】
"""
    
    for method, count in method_counts.items():
        percentage = (count / total_count * 100) if total_count > 0 else 0
        report_text += f"  {method}: {count}人 ({percentage:.1f}%)\n"
    
    report_text += f"""
【成绩模式统计】
"""
    
    for pattern, count in pattern_counts.items():
        percentage = (count / total_count * 100) if total_count > 0 else 0
        report_text += f"  {pattern}: {count}人 ({percentage:.1f}%)\n"
    
    report_text += f"""
【变化趋势统计】
"""
    
    for trend, count in trend_counts.items():
        percentage = (count / total_count * 100) if total_count > 0 else 0
        report_text += f"  {trend}: {count}人 ({percentage:.1f}%)\n"
    
    report_text += f"""
【教学建议】
"""
    
    # 基于模式给出建议
    if '强下降趋势' in pattern_counts or '近期下降' in pattern_counts:
        down_count = pattern_counts.get('强下降趋势', 0) + pattern_counts.get('近期下降', 0)
        report_text += f"1. 重点关注 {down_count} 名成绩下降学生，提供个性化辅导\n"
    
    if '强上升趋势' in pattern_counts or '近期上升' in pattern_counts:
        up_count = pattern_counts.get('强上升趋势', 0) + pattern_counts.get('近期上升', 0)
        report_text += f"2. 鼓励 {up_count} 名成绩上升学生，保持学习动力\n"
    
    if '稳定' in pattern_counts or '低波动稳定' in pattern_counts:
        stable_count = pattern_counts.get('稳定', 0) + pattern_counts.get('低波动稳定', 0)
        report_text += f"3. {stable_count} 名学生成绩稳定，维持现有教学节奏\n"
    
    if '高波动' in pattern_counts:
        volatile_count = pattern_counts.get('高波动', 0)
        report_text += f"4. {volatile_count} 名学生成绩波动大，需分析原因并提供稳定性训练\n"
    
    report_text += f"""
【学生详细预测结果】
序号,姓名,当前成绩,预测成绩,预测变化,变化趋势,成绩模式,预测方法,置信度,有效考试次数,数据充足度
"""
    
    # 添加每个学生的详细结果
    for _, row in results_df.iterrows():
        report_text += f"{row['序号']},{row['姓名']},{row['当前成绩']},{row['预测成绩']},"
        report_text += f"{row['预测变化']},{row['变化趋势']},{row['成绩模式']},"
        report_text += f"{row['预测方法']},{row['置信度']},{row['有效考试次数']},{row['数据充足度']}\n"
    

    report_text += """

【智能预测说明】
1. 趋势感知预测：分析学生历史成绩模式，识别上升/下降趋势
2. 迁移学习：参考相似学生的成绩变化模式进行预测
3. 聚类分析：将学生按成绩模式分组，提供更精准的预测
4. 置信度计算：综合考虑数据量、模式稳定性和相似度
5. 边界检查：确保预测成绩在合理范围内

======================================
"""
    
    return report_text


# ============================================
# 模块4: 学生成绩分析、预测（智能版）主函数
# 结合趋势感知、迁移学习和聚类分析
# ============================================
def module_student_analysis():
    """模块4: 学生成绩分析、预测（智能版）"""
    st.markdown("## 📈 4. 学生成绩分析、预测")
    
    if not st.session_state.data_loaded:
        st.warning("请先上传数据文件（切换到'导入数据文件'模块）")
        return
    
    st.markdown("""
    **智能版功能特色**：
    1. **趋势感知预测**：深度分析学生成绩模式，识别上升/下降趋势
    2. **迁移学习**：参考相似学生的学习轨迹进行智能预测
    3. **聚类分析**：将学生按成绩模式分组，提供个性化预测
    4. **智能批量预测**：支持多种预测模式和高级分析
    5. **异常检测**：识别异常成绩波动
    6. **科目关联分析**：分析科目间相关性
    """)
    
    # 选择分析类型
    st.markdown("### 📊 选择分析类型")
    analysis_type = st.selectbox(
        "分析类型",
        ["成绩趋势分析（智能版）", "智能成绩预测", "智能批量成绩预测", "成绩异常检测", "科目关联分析"],
        key="analysis_type_select"
    )
    
    if analysis_type == "成绩趋势分析（智能版）":
        st.markdown("#### 📈 智能版成绩趋势分析")
        st.markdown("深度分析学生成绩模式，识别长期和短期趋势。")
        
        # 选择班级和学生
        classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
        
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox(
                "选择班级",
                classes,
                key="trend_analysis_class_select"
            )
        
        with col2:
            if selected_class:
                class_students = st.session_state.df[
                    st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == selected_class
                ][st.session_state.name_column_name].dropna().unique()
                
                if len(class_students) > 0:
                    selected_student = st.selectbox(
                        "选择学生",
                        sorted(class_students),
                        key="trend_analysis_student_select"
                    )
                else:
                    st.warning("该班级没有学生数据")
                    selected_student = None
            else:
                selected_student = None
        
        if selected_student:
            # 获取学生成绩
            grades_df = GradeManager.get_student_grades(
                st.session_state.df, selected_class, selected_student,
                st.session_state.class_column_name, st.session_state.name_column_name,
                st.session_state.subjects, st.session_state.exams, 
                st.session_state.column_mapping
            )
            
            if grades_df is not None and not grades_df.empty:
                st.success(f"✅ 成功获取 {selected_class} - {selected_student} 的成绩数据")
                
                # 选择要分析的科目
                score_subjects = [s for s in st.session_state.subjects if '排' not in s and s in grades_df.columns]
                
                if score_subjects:
                    selected_subject = st.selectbox(
                        "选择要分析的科目",
                        score_subjects,
                        key="trend_analysis_subject"
                    )
                    
                    if selected_subject and selected_subject in grades_df.columns:
                        # 提取成绩数据
                        subject_data = pd.to_numeric(grades_df[selected_subject], errors='coerce')
                        
                        # 创建一个布尔掩码，标识非空值
                        valid_mask = subject_data.notna()
                        
                        # 获取有效成绩
                        valid_data = subject_data[valid_mask]
                        
                        # 获取对应的考试场次
                        exam_names = grades_df.loc[valid_mask, '考试场次'].tolist()
                        
                        if len(valid_data) >= 3:  # 至少需要3个数据点
                            st.markdown("#### 🧠 智能模式分析")
                            
                            # 检查数据一致性
                            if len(valid_data) != len(exam_names):
                                st.error(f"数据不一致：成绩数量({len(valid_data)}) ≠ 考试场次数量({len(exam_names)})")
                                st.warning("可能存在数据质量问题，请检查原始数据")
                                return
                            
                            # 创建智能预测器
                            predictor = IntelligentGradePredictor()
                            
                            # 进行分析
                            with st.spinner("正在进行智能模式分析..."):
                                # 分析学生模式
                                pattern_info = predictor.analyze_student_pattern(valid_data.values)
                                
                                # 使用智能预测
                                prediction_result = predictor.smart_predict(valid_data.values, selected_student)
                            
                            # 显示模式分析结果
                            st.markdown("##### 🎯 成绩模式分析")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                pattern_type = pattern_info.get("pattern_type", "unknown")
                                pattern_map = {
                                    "strong_upward": "强上升趋势",
                                    "recent_upward": "近期上升",
                                    "strong_downward": "强下降趋势", 
                                    "recent_downward": "近期下降",
                                    "stable": "稳定",
                                    "stable_low_volatility": "低波动稳定",
                                    "volatile": "高波动",
                                    "changed_at": "模式转变"
                                }
                                pattern_desc = pattern_map.get(pattern_type, pattern_type)
                                pattern_emoji = "📈" if "upward" in pattern_type else "📉" if "downward" in pattern_type else "➡️"
                                st.metric("成绩模式", f"{pattern_desc} {pattern_emoji}",
                                         f"模式强度: {pattern_info.get('trend_strength', 0):.2f}")
                            
                            with col2:
                                slope = pattern_info.get("slope", 0)
                                trend = "上升" if slope > 0.1 else "下降" if slope < -0.1 else "平稳"
                                trend_emoji = "📈" if slope > 0.1 else "📉" if slope < -0.1 else "➡️"
                                st.metric("长期趋势", f"{trend} {trend_emoji}",
                                         f"斜率: {slope:.3f}")
                            
                            with col3:
                                recent_slope = pattern_info.get("recent_slope", 0)
                                recent_trend = "上升" if recent_slope > 0.2 else "下降" if recent_slope < -0.2 else "平稳"
                                st.metric("近期趋势", recent_trend,
                                         f"近期斜率: {recent_slope:.3f}")
                            
                            # 显示统计数据
                            st.markdown("##### 📊 统计信息")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("历史平均分", f"{pattern_info.get('mean', 0):.1f}",
                                         f"最新成绩: {valid_data.values[-1]:.1f}")
                            
                            with col2:
                                std_score = pattern_info.get("std", 0)
                                volatility = "高" if std_score > 10 else "中" if std_score > 5 else "低"
                                st.metric("成绩波动性", volatility,
                                         f"标准差: {std_score:.2f}")
                            
                            with col3:
                                st.metric("有效考试次数", len(valid_data),
                                         f"数据充足" if len(valid_data) >= 5 else "数据适中")
                            
                            # 创建趋势图表
                            st.markdown("##### 📈 可视化分析")
                            
                            # 创建一个过滤后的DataFrame，只包含有效成绩
                            valid_grades_df = grades_df.loc[valid_mask].copy()
                            
                            # 创建趋势图表
                            fig = ChartGenerator.create_grade_trend_chart(
                                valid_grades_df, [selected_subject], selected_student, selected_class
                            )
                            
                            if fig:
                                # 添加趋势线
                                time_indices = np.arange(len(valid_grades_df['考试场次']))
                                slope = pattern_info.get("slope", 0)
                                intercept = pattern_info.get("mean", 0) - slope * (len(valid_grades_df['考试场次'])-1)/2
                                y_vals = intercept + slope * time_indices
                                
                                # 添加预测点
                                if 'prediction' in prediction_result:
                                    x_pred = [len(valid_grades_df['考试场次'])]
                                    y_pred = [prediction_result['prediction']]
                                    
                                    # 缩小置信区间 - 从5改为2.5
                                    confidence_interval = 2.5
                                    
                                    # 创建趋势线
                                    fig.add_trace(go.Scatter(
                                        x=valid_grades_df['考试场次'],
                                        y=y_vals,
                                        mode='lines',
                                        name='长期趋势线',
                                        line=dict(color='red', width=2, dash='dash')
                                    ))
                                    
                                    # 创建预测点（星星标记）
                                    fig.add_trace(go.Scatter(
                                        x=[f'预测{len(valid_grades_df)+1}'],
                                        y=y_pred,
                                        mode='markers',
                                        name='智能预测',
                                        marker=dict(size=12, color='green', symbol='star'),
                                        error_y=dict(
                                            type='data',
                                            array=[confidence_interval],  # 使用较小的置信区间
                                            visible=True
                                        )
                                    ))
                                
                                st.plotly_chart(fig, use_container_width=True, key="intelligent_trend_chart")
                            
                            # 显示预测结果
                            st.markdown("##### 🔮 智能预测结果")
                            if 'prediction' in prediction_result:
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "下次考试预测", 
                                        f"{prediction_result['prediction']:.1f}",
                                        f"基于{pattern_desc}模式"
                                    )
                                
                                with col2:
                                    confidence_score = prediction_result.get('confidence_score', 0.5)
                                    confidence_level = "高" if confidence_score > 0.7 else "中" if confidence_score > 0.5 else "低"
                                    st.metric("预测置信度", f"{confidence_score:.1%}",
                                             f"{confidence_level}置信度")
                                
                                with col3:
                                    current_grade = valid_data.values[-1]
                                    improvement = prediction_result['prediction'] - current_grade
                                    trend = "上升" if improvement > 0 else "下降"
                                    st.metric("预计变化", f"{trend} {abs(improvement):.1f}分",
                                             f"与当前成绩相比")
                            
                            # 显示模式详情
                            st.markdown("##### 🔍 模式详情")
                            
                            with st.expander("查看详细模式分析"):
                                st.markdown(f"""
                                **模式类型**: {pattern_desc}
                                
                                **长期趋势斜率**: {pattern_info.get('slope', 0):.3f}
                                
                                **近期趋势斜率**: {pattern_info.get('recent_slope', 0):.3f}
                                
                                **趋势强度**: {pattern_info.get('trend_strength', 0):.3f}
                                
                                **平均成绩**: {pattern_info.get('mean', 0):.1f}
                                
                                **成绩标准差**: {pattern_info.get('std', 0):.2f}
                                
                                **模式变化**: {pattern_info.get('has_pattern_change', False)}
                                
                                **变化点**: {pattern_info.get('change_point', '无')}
                                
                                **数据点数**: {len(valid_data)}
                                """)
                            
                            # 显示预测方法说明
                            st.markdown("##### 📖 预测方法说明")
                            
                            if 'transfer_learning' in prediction_result and prediction_result['transfer_learning']:
                                st.success(f"""
                                **✅ 使用融合预测方法**
                                
                                **趋势感知预测**: {prediction_result.get('base_prediction', 0):.1f}
                                
                                **迁移学习调整**: {prediction_result.get('transfer_prediction', 0):.1f}
                                
                                **相似学生数**: {prediction_result.get('similar_students_count', 0)}
                                
                                **平均相似度**: {prediction_result.get('avg_similarity', 0):.1%}
                                
                                **最终预测**: 趋势感知(70%) + 迁移学习(30%)
                                """)
                            else:
                                st.info(f"""
                                **ℹ️ 使用趋势感知预测**
                                
                                **基础预测**: {prediction_result.get('base_prediction', 0):.1f}
                                
                                **趋势调整**: {prediction_result.get('trend_adjustment', 0):.1f}
                                
                                **最终预测**: 基础预测 + 趋势调整
                                
                                **预测方法**: 自适应加权平均 + 趋势调整
                                """)
                            
                            # 显示缺失数据信息
                            missing_count = len(subject_data) - len(valid_data)
                            if missing_count > 0:
                                st.warning(f"⚠️ 注意：该科目有 {missing_count} 个缺失值，已自动过滤。分析基于 {len(valid_data)} 个有效成绩。")
                            
                            # 显示学习建议
                            st.markdown("##### 💡 智能学习建议")
                            
                            suggestions = generate_intelligent_suggestions(pattern_info, prediction_result, valid_data.values[-1])
                            for suggestion in suggestions:
                                st.info(suggestion)
                            
                        else:
                            st.warning(f"需要至少3次有效成绩才能进行智能分析，当前只有{len(valid_data)}次")
                else:
                    st.info("没有找到可分析的成绩科目")
            else:
                st.error(f"❌ 未找到学生 {selected_class} - {selected_student} 的成绩数据")
    
    elif analysis_type == "智能成绩预测":
        st.markdown("#### 🔮 智能成绩预测")
        st.markdown("结合趋势感知、迁移学习和聚类分析的智能预测。")
        
        # 选择班级和学生
        classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
        
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox(
                "选择班级",
                classes,
                key="intelligent_prediction_class_select"
            )
        
        with col2:
            if selected_class:
                class_students = st.session_state.df[
                    st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == selected_class
                ][st.session_state.name_column_name].dropna().unique()
                
                if len(class_students) > 0:
                    selected_student = st.selectbox(
                        "选择学生",
                        sorted(class_students),
                        key="intelligent_prediction_student_select"
                    )
                else:
                    st.warning("该班级没有学生数据")
                    selected_student = None
            else:
                selected_student = None
        
        if selected_student:
            # 获取学生成绩
            grades_df = GradeManager.get_student_grades(
                st.session_state.df, selected_class, selected_student,
                st.session_state.class_column_name, st.session_state.name_column_name,
                st.session_state.subjects, st.session_state.exams, 
                st.session_state.column_mapping
            )
            
            if grades_df is not None and not grades_df.empty:
                st.success(f"✅ 成功获取 {selected_class} - {selected_student} 的成绩数据")
                
                # 选择要预测的科目
                score_subjects = [s for s in st.session_state.subjects if '排' not in s and s in grades_df.columns]
                
                if score_subjects:
                    selected_subject = st.selectbox(
                        "选择要预测的科目",
                        score_subjects,
                        key="intelligent_prediction_subject"
                    )
                    
                    if selected_subject and selected_subject in grades_df.columns:
                        # 提取成绩数据
                        subject_data = pd.to_numeric(grades_df[selected_subject], errors='coerce')
                        
                        # 创建一个布尔掩码，标识非空值
                        valid_mask = subject_data.notna()
                        
                        # 获取有效成绩
                        valid_data = subject_data[valid_mask]
                        
                        # 获取对应的考试场次
                        exam_names = grades_df.loc[valid_mask, '考试场次'].tolist()
                        
                        if len(valid_data) >= 2:  # 至少需要2个数据点
                            st.markdown("#### 🤖 智能预测系统")
                            
                            # 检查数据一致性
                            if len(valid_data) != len(exam_names):
                                st.error(f"数据不一致：成绩数量({len(valid_data)}) ≠ 考试场次数量({len(exam_names)})")
                                st.warning("可能存在数据质量问题，请检查原始数据")
                                return
                            
                            # 收集全班数据用于迁移学习
                            with st.spinner("收集班级数据用于迁移学习..."):
                                all_students_scores = {}
                                
                                for student in class_students:
                                    student_grades_df = GradeManager.get_student_grades(
                                        st.session_state.df, selected_class, student,
                                        st.session_state.class_column_name, st.session_state.name_column_name,
                                        st.session_state.subjects, st.session_state.exams, 
                                        st.session_state.column_mapping
                                    )
                                    
                                    if student_grades_df is not None and selected_subject in student_grades_df.columns:
                                        scores = pd.to_numeric(student_grades_df[selected_subject], errors='coerce').dropna().values
                                        if len(scores) >= 2:
                                            all_students_scores[student] = scores
                            
                            with st.spinner("正在进行智能预测..."):
                                # 创建智能预测器
                                predictor = IntelligentGradePredictor(all_students_scores)
                                
                                # 进行智能预测
                                prediction_result = predictor.smart_predict(
                                    valid_data.values, selected_student, all_students_scores
                                )
                                
                                if not prediction_result:
                                    st.error("预测失败，无法生成预测结果")
                                    return
                                
                                # 分析模式
                                pattern_info = predictor.analyze_student_pattern(valid_data.values)
                            
                            # 显示数据信息
                            st.markdown("##### 📊 数据信息")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("有效成绩数", len(valid_data))
                            
                            with col2:
                                data_quality = "充足" if len(valid_data) >= 6 else "中等" if len(valid_data) >= 3 else "较少"
                                st.metric("数据量", data_quality)
                            
                            with col3:
                                st.metric("班级学生数", len(all_students_scores))
                            
                            # 显示预测结果
                            st.markdown("##### 🔮 预测结果")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(
                                    "下次考试预测", 
                                    f"{prediction_result['prediction']:.1f}",
                                    f"基于{prediction_result.get('method', '智能预测')}"
                                )
                            
                            with col2:
                                confidence_score = prediction_result.get('confidence_score', 0.5)
                                confidence_level = "高" if confidence_score > 0.7 else "中" if confidence_score > 0.5 else "低"
                                st.metric(
                                    "预测置信度", 
                                    f"{confidence_score:.1%} ({confidence_level})",
                                    f"模型可靠性"
                                )
                            
                            with col3:
                                current_grade = valid_data.values[-1]
                                improvement = prediction_result['prediction'] - current_grade
                                trend = "上升" if improvement > 0 else "下降"
                                st.metric(
                                    "预计变化", 
                                    f"{trend} {abs(improvement):.1f}分",
                                    f"与当前成绩相比"
                                )
                            
                            # 显示模式信息
                            st.markdown("##### 🎯 成绩模式")
                            pattern_type = prediction_result.get('pattern', 'unknown')
                            pattern_map = {
                                "strong_upward": "强上升趋势",
                                "recent_upward": "近期上升", 
                                "strong_downward": "强下降趋势",
                                "recent_downward": "近期下降",
                                "stable": "稳定",
                                "stable_low_volatility": "低波动稳定",
                                "volatile": "高波动"
                            }
                            pattern_desc = pattern_map.get(pattern_type, pattern_type)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("模式类型", pattern_desc)
                            with col2:
                                st.metric("趋势强度", f"{prediction_result.get('trend_strength', 0):.2f}")
                            with col3:
                                st.metric("斜率", f"{prediction_result.get('slope', 0):.3f}")
                            
                            # 显示预测详情
                            st.markdown("##### 📈 预测详情")
                            
                            with st.expander("🔍 查看预测详情"):
                                st.markdown(f"""
                                **预测方法**: {prediction_result.get('method', '未知')}
                                
                                **基础预测值**: {prediction_result.get('base_prediction', 0):.1f}
                                
                                **趋势调整值**: {prediction_result.get('trend_adjustment', 0):.1f}
                                
                                **置信区间**: ±{prediction_result.get('confidence_interval', 5):.1f}
                                
                                **数据点数**: {len(valid_data)}
                                """)
                                
                                if prediction_result.get('transfer_learning', False):
                                    st.markdown(f"""
                                    **迁移学习信息**:
                                    - 相似学生数: {prediction_result.get('similar_students_count', 0)}
                                    - 平均相似度: {prediction_result.get('avg_similarity', 0):.1%}
                                    - 迁移预测值: {prediction_result.get('transfer_prediction', 0):.1f}
                                    """)
                            
                            # 可视化预测结果
                            st.markdown("##### 📈 预测可视化")
                            
                            # 创建一个过滤后的DataFrame，只包含有效成绩
                            valid_grades_df = grades_df.loc[valid_mask].copy()
                            
                            # 创建趋势图表
                            fig = ChartGenerator.create_grade_trend_chart(
                                valid_grades_df, [selected_subject], selected_student, selected_class
                            )
                            
                            if fig:
                                # 添加趋势线
                                time_indices = np.arange(len(valid_grades_df['考试场次']))
                                slope = pattern_info.get("slope", 0)
                                intercept = pattern_info.get("mean", 0) - slope * (len(valid_grades_df['考试场次'])-1)/2
                                y_vals = intercept + slope * time_indices
                                
                                # 添加预测点
                                x_pred = [len(valid_grades_df['考试场次'])]
                                y_pred = [prediction_result['prediction']]
                                
                                # 缩小置信区间 - 从5改为2.5
                                confidence_interval = 2.5
                                
                                # 创建趋势线
                                fig.add_trace(go.Scatter(
                                    x=valid_grades_df['考试场次'],
                                    y=y_vals,
                                    mode='lines',
                                    name='长期趋势线',
                                    line=dict(color='red', width=2, dash='dash')
                                ))
                                
                                # 创建预测点（星星标记）
                                fig.add_trace(go.Scatter(
                                    x=[f'智能预测{len(valid_grades_df)+1}'],
                                    y=y_pred,
                                    mode='markers',
                                    name=f"{prediction_result.get('method', '智能预测')}",
                                    marker=dict(size=12, color='green', symbol='star'),
                                    error_y=dict(
                                        type='data',
                                        array=[confidence_interval],  # 使用较小的置信区间
                                        visible=True
                                    ),
                                    hovertemplate=f'预测: %{{y:.1f}}<br>' +
                                                 f'置信区间: ±{confidence_interval:.1f}<br>' +
                                                 f'方法: {prediction_result.get("method", "智能预测")}<extra></extra>'
                                ))
                                
                                st.plotly_chart(fig, use_container_width=True, key="intelligent_prediction_chart")
                            
                            # 显示相似学生（如果使用了迁移学习）
                            if prediction_result.get('transfer_learning', False) and all_students_scores:
                                st.markdown("##### 👥 相似学生参考")
                                
                                # 寻找相似学生
                                similar_students = predictor.find_similar_students(
                                    valid_data.values, all_students_scores, n_similar=3
                                )
                                
                                if similar_students:
                                    st.info("**参考以下相似学生的学习轨迹：**")
                                    
                                    for i, (student_name, similarity) in enumerate(similar_students[:3], 1):
                                        if student_name in all_students_scores:
                                            scores = all_students_scores[student_name]
                                            if len(scores) > len(valid_data):
                                                next_score = scores[len(valid_data)]  # 对应位置的下次成绩
                                                
                                                with st.expander(f"相似学生 {i}: {student_name} (相似度: {similarity:.1%})"):
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        st.metric("当前成绩", f"{scores[-1]:.1f}")
                                                    with col2:
                                                        st.metric("下次成绩", f"{next_score:.1f}")
                                                    with col3:
                                                        change = next_score - scores[-1]
                                                        st.metric("实际变化", f"{change:+.1f}")
                            
                            # 显示学习建议
                            st.markdown("##### 💡 智能学习建议")
                            
                            suggestions = []
                            
                            if prediction_result['prediction'] > current_grade:
                                suggestions.append(f"🔮 **预测成绩上升**：预计下次考试成绩为{prediction_result['prediction']:.1f}分，比当前提高{improvement:.1f}分。")
                            elif prediction_result['prediction'] < current_grade:
                                suggestions.append(f"⚠️ **预测成绩下降**：预计下次考试成绩为{prediction_result['prediction']:.1f}分，比当前下降{abs(improvement):.1f}分，需要加强复习。")
                            
                            if pattern_type in ["strong_upward", "recent_upward"]:
                                suggestions.append("📈 **保持上升趋势**：当前呈现上升趋势，继续保持当前的学习方法和努力程度。")
                            elif pattern_type in ["strong_downward", "recent_downward"]:
                                suggestions.append("📉 **注意下降趋势**：近期成绩有下降趋势，建议分析原因并调整学习策略。")
                            elif pattern_type == "stable":
                                suggestions.append("➡️ **成绩稳定**：成绩保持稳定，可以尝试挑战更高目标。")
                            elif pattern_type == "volatile":
                                suggestions.append("🎢 **成绩波动大**：成绩波动较大，建议分析波动原因，提高稳定性。")
                            
                            if len(valid_data) < 5:
                                suggestions.append("📊 **建议收集更多数据**：更多考试成绩数据可以提高预测准确性。")
                            
                            for suggestion in suggestions:
                                st.info(suggestion)
                            
                        else:
                            st.warning(f"需要至少2次有效成绩才能进行预测，当前只有{len(valid_data)}次")
                else:
                    st.info("没有找到可预测的成绩科目")
            else:
                st.error(f"❌ 未找到学生 {selected_class} - {selected_student} 的成绩数据")
    
    elif analysis_type == "智能批量成绩预测":
        # 调用智能批量预测功能
        intelligent_batch_prediction()
    
    elif analysis_type == "成绩异常检测":
        st.markdown("#### ⚠️ 成绩异常检测")
        st.markdown("检测学生成绩中的异常波动，帮助发现学习问题。")
        
        # 选择班级和学生
        classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
        
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox(
                "选择班级",
                classes,
                key="anomaly_class_select"
            )
        
        with col2:
            if selected_class:
                class_students = st.session_state.df[
                    st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == selected_class
                ][st.session_state.name_column_name].dropna().unique()
                
                if len(class_students) > 0:
                    selected_student = st.selectbox(
                        "选择学生",
                        sorted(class_students),
                        key="anomaly_student_select"
                    )
                else:
                    st.warning("该班级没有学生数据")
                    selected_student = None
            else:
                selected_student = None
        
        if selected_student:
            # 获取学生成绩
            grades_df = GradeManager.get_student_grades(
                st.session_state.df, selected_class, selected_student,
                st.session_state.class_column_name, st.session_state.name_column_name,
                st.session_state.subjects, st.session_state.exams, 
                st.session_state.column_mapping
            )
            
            if grades_df is not None and not grades_df.empty:
                st.success(f"✅ 成功获取 {selected_class} - {selected_student} 的成绩数据")
                
                # 选择要检测的科目
                score_subjects = [s for s in st.session_state.subjects if '排' not in s and s in grades_df.columns]
                
                if score_subjects:
                    selected_subjects = st.multiselect(
                        "选择要检测的科目",
                        score_subjects,
                        default=score_subjects[:min(3, len(score_subjects))],
                        key="anomaly_subjects_select"
                    )
                    
                    if selected_subjects:
                        st.markdown("#### 📊 异常检测结果")
                        
                        anomalies = []
                        
                        for subject in selected_subjects:
                            if subject in grades_df.columns:
                                scores = pd.to_numeric(grades_df[subject], errors='coerce')
                                
                                # 确保使用有效成绩
                                valid_mask = scores.notna()
                                valid_scores = scores[valid_mask]
                                valid_exams = grades_df.loc[valid_mask, '考试场次']
                                
                                if len(valid_scores) >= 3:
                                    # 计算Z-score
                                    mean_score = valid_scores.mean()
                                    std_score = valid_scores.std()
                                    
                                    if std_score > 0:  # 避免除零
                                        z_scores = (valid_scores - mean_score) / std_score
                                        
                                        # 检测异常（|Z| > 2）
                                        anomaly_indices = np.where(np.abs(z_scores) > 2)[0]
                                        
                                        for idx in anomaly_indices:
                                            exam_name = valid_exams.iloc[idx]
                                            actual_score = valid_scores.iloc[idx]
                                            z_score = z_scores.iloc[idx]
                                            
                                            anomalies.append({
                                                '科目': subject,
                                                '考试场次': exam_name,
                                                '成绩': actual_score,
                                                'Z分数': z_score,
                                                '异常类型': '过高' if z_score > 0 else '过低'
                                            })
                        
                        if anomalies:
                            st.warning(f"⚠️ 检测到 {len(anomalies)} 个异常成绩")
                            
                            anomalies_df = pd.DataFrame(anomalies)
                            st.dataframe(anomalies_df, use_container_width=True)
                            
                            st.markdown("#### 📈 异常成绩可视化")
                            
                            fig = go.Figure()
                            
                            for subject in selected_subjects:
                                if subject in grades_df.columns:
                                    scores = pd.to_numeric(grades_df[subject], errors='coerce')
                                    
                                    # 确保使用有效成绩
                                    valid_mask = scores.notna()
                                    valid_scores = scores[valid_mask]
                                    valid_exams = grades_df.loc[valid_mask, '考试场次']
                                    
                                    # 添加正常成绩
                                    normal_mask = ~valid_scores.index.isin([a.get('index', -1) for a in anomalies if a['科目'] == subject])
                                    fig.add_trace(go.Scatter(
                                        x=valid_exams[normal_mask],
                                        y=valid_scores[normal_mask],
                                        mode='lines+markers',
                                        name=f'{subject} (正常)',
                                        line=dict(width=2),
                                        marker=dict(size=6)
                                    ))
                            
                            # 添加异常成绩
                            for anomaly in anomalies:
                                fig.add_trace(go.Scatter(
                                    x=[anomaly['考试场次']],
                                    y=[anomaly['成绩']],
                                    mode='markers',
                                    name=f"{anomaly['科目']} (异常)",
                                    marker=dict(
                                        size=12,
                                        color='red' if anomaly['异常类型'] == '过高' else 'orange',
                                        symbol='x' if anomaly['异常类型'] == '过高' else 'triangle-down'
                                    ),
                                    text=f"Z分数: {anomaly['Z分数']:.2f}",
                                    hovertemplate='<b>%{text}</b><br>考试场次: %{x}<br>成绩: %{y}<extra></extra>'
                                ))
                            
                            fig.update_layout(
                                title=f"{selected_class} - {selected_student} 成绩异常检测",
                                xaxis_title='考试场次',
                                yaxis_title='成绩',
                                height=500,
                                template='plotly_white',
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key="anomaly_chart")
                            
                            st.info("""
                            **异常检测标准**：
                            1. 使用Z-score方法
                            2. |Z| > 2 视为异常
                            3. 红色×表示过高异常
                            4. 橙色▼表示过低异常
                            """)
                        else:
                            st.success("✅ 未检测到异常成绩，成绩波动在正常范围内")
                else:
                    st.info("没有找到可分析的成绩科目")
            else:
                st.error(f"❌ 未找到学生 {selected_class} - {selected_student} 的成绩数据")
    
    elif analysis_type == "科目关联分析":
        st.markdown("#### 🔗 科目关联分析")
        st.markdown("分析各科目成绩之间的关联性，发现优势科目和弱势科目。")
        
        # 选择班级和学生
        classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
        
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox(
                "选择班级",
                classes,
                key="correlation_class_select"
            )
        
        with col2:
            if selected_class:
                class_students = st.session_state.df[
                    st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == selected_class
                ][st.session_state.name_column_name].dropna().unique()
                
                if len(class_students) > 0:
                    selected_student = st.selectbox(
                        "选择学生",
                        sorted(class_students),
                        key="correlation_student_select"
                    )
                else:
                    st.warning("该班级没有学生数据")
                    selected_student = None
            else:
                selected_student = None
        
        if selected_student:
            # 获取学生成绩
            grades_df = GradeManager.get_student_grades(
                st.session_state.df, selected_class, selected_student,
                st.session_state.class_column_name, st.session_state.name_column_name,
                st.session_state.subjects, st.session_state.exams, 
                st.session_state.column_mapping
            )
            
            if grades_df is not None and not grades_df.empty:
                st.success(f"✅ 成功获取 {selected_class} - {selected_student} 的成绩数据")
                
                # 选择要分析的科目
                score_subjects = [s for s in st.session_state.subjects if '排' not in s and s in grades_df.columns]
                
                if len(score_subjects) >= 2:
                    selected_subjects = st.multiselect(
                        "选择要分析的科目（至少2个）",
                        score_subjects,
                        default=score_subjects[:min(4, len(score_subjects))],
                        key="correlation_subjects_select"
                    )
                    
                    if len(selected_subjects) >= 2:
                        st.markdown("#### 📊 科目关联分析")
                        
                        # 确保所有科目在相同考试场次上都有成绩
                        valid_mask = None
                        for subject in selected_subjects:
                            if subject in grades_df.columns:
                                subject_mask = grades_df[subject].notna()
                                if valid_mask is None:
                                    valid_mask = subject_mask
                                else:
                                    valid_mask = valid_mask & subject_mask
                        
                        if valid_mask is not None and valid_mask.any():
                            # 只保留所有科目都有成绩的考试场次
                            valid_grades_df = grades_df[valid_mask]
                            
                            # 计算相关性矩阵
                            correlation_data = []
                            for subject in selected_subjects:
                                if subject in valid_grades_df.columns:
                                    correlation_data.append(pd.to_numeric(valid_grades_df[subject], errors='coerce'))
                            
                            if len(correlation_data) >= 2:
                                correlation_df = pd.DataFrame(correlation_data, index=selected_subjects).T
                                correlation_matrix = correlation_df.corr()
                                
                                # 显示增强版热力图
                                st.markdown("##### 📈 科目相关性热力图")
                                
                                correlation_values = correlation_matrix.values
                                
                                # 创建热力图
                                fig = go.Figure(data=go.Heatmap(
                                    z=correlation_values,
                                    x=correlation_matrix.columns,
                                    y=correlation_matrix.index,
                                    colorscale='RdBu',
                                    zmin=-1,
                                    zmax=1,
                                    hoverongaps=False,
                                    hoverinfo='x+y+z',
                                    zmid=0,
                                ))
                                
                                # 为每个单元格添加annotation
                                annotations = []
                                for i in range(len(correlation_matrix.index)):
                                    for j in range(len(correlation_matrix.columns)):
                                        value = correlation_values[i, j]
                                        
                                        # 根据相关性绝对值决定字体颜色
                                        if abs(value) > 0.5:
                                            font_color = 'white'
                                        else:
                                            font_color = 'black'
                                        
                                        annotation = dict(
                                            x=correlation_matrix.columns[j],
                                            y=correlation_matrix.index[i],
                                            text=f"{value:.3f}",
                                            showarrow=False,
                                            font=dict(
                                                size=16,
                                                family="Arial, sans-serif",
                                                color=font_color
                                            )
                                        )
                                        annotations.append(annotation)
                                
                                fig.update_layout(
                                    title=dict(
                                        text=f"{selected_class} - {selected_student} 科目成绩相关性",
                                        font=dict(size=20, family="Arial, sans-serif", color='#333333'),
                                        x=0.5,
                                        xanchor='center'
                                    ),
                                    xaxis_title='科目',
                                    yaxis_title='科目',
                                    height=500,
                                    width=600,
                                    template='plotly_white',
                                    font=dict(size=14, family="Arial, sans-serif"),
                                    xaxis=dict(
                                        tickfont=dict(size=14, family="Arial, sans-serif"),
                                        title_font=dict(size=16, family="Arial, sans-serif"),
                                        showgrid=True,
                                        gridwidth=0.5,
                                        gridcolor='lightgray'
                                    ),
                                    yaxis=dict(
                                        tickfont=dict(size=14, family="Arial, sans-serif"),
                                        title_font=dict(size=16, family="Arial, sans-serif"),
                                        showgrid=True,
                                        gridwidth=0.5,
                                        gridcolor='lightgray'
                                    ),
                                    coloraxis_colorbar=dict(
                                        title="相关系数",
                                        title_font=dict(size=14),
                                        tickfont=dict(size=12),
                                        thickness=20,
                                        len=0.8,
                                        yanchor="middle",
                                        y=0.5
                                    ),
                                    margin=dict(l=80, r=80, t=100, b=80),
                                    paper_bgcolor='rgba(240, 240, 240, 0.1)',
                                    plot_bgcolor='rgba(255, 255, 255, 0.9)',
                                    annotations=annotations
                                )
                                
                                fig.update_traces(
                                    showscale=True,
                                    hovertemplate='<b>%{x}</b> 与 <b>%{y}</b><br>相关性: %{z:.3f}<extra></extra>',
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, key="correlation_heatmap")
                                
                                # 分析结果
                                st.markdown("##### 📈 关联分析结果")
                                
                                # 找出相关性最高的科目对
                                corr_values = []
                                for i in range(len(correlation_matrix.columns)):
                                    for j in range(i+1, len(correlation_matrix.columns)):
                                        subject1 = correlation_matrix.columns[i]
                                        subject2 = correlation_matrix.columns[j]
                                        corr_value = correlation_matrix.iloc[i, j]
                                        corr_values.append((subject1, subject2, corr_value))
                                
                                if corr_values:
                                    # 按相关性绝对值排序
                                    corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
                                    
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.markdown("**🔗 最强正相关**")
                                        positive_corr = [cv for cv in corr_values if cv[2] > 0]
                                        if positive_corr:
                                            best_positive = positive_corr[0]
                                            st.metric(
                                                f"{best_positive[0]} - {best_positive[1]}",
                                                f"{best_positive[2]:.3f}",
                                                "强正相关"
                                            )
                                            st.info("这两门科目成绩变化趋势一致，一科好另一科也好")
                                    
                                    with col2:
                                        st.markdown("**🔄 最强负相关**")
                                        negative_corr = [cv for cv in corr_values if cv[2] < 0]
                                        if negative_corr:
                                            best_negative = negative_corr[0]
                                            st.metric(
                                                f"{best_negative[0]} - {best_negative[1]}",
                                                f"{best_negative[2]:.3f}",
                                                "强负相关"
                                            )
                                            st.warning("这两门科目成绩变化趋势相反，一科好另一科可能差")
                                    
                                    with col3:
                                        st.markdown("**📊 相关性统计**")
                                        positive_count = len(positive_corr)
                                        negative_count = len(negative_corr)
                                        neutral_count = len(corr_values) - positive_count - negative_count
                                        st.metric(
                                            "相关类型分布",
                                            f"{positive_count}正/{negative_count}负",
                                            f"共{len(corr_values)}对"
                                        )
                                        st.info(f"正相关: {positive_count}, 负相关: {negative_count}")
                                    
                                    # 显示详细的相关性表格
                                    with st.expander("📋 查看详细的科目相关性矩阵"):
                                        st.markdown("**数值表格**")
                                        styled_df = correlation_matrix.style.format("{:.3f}")
                                        st.dataframe(styled_df, use_container_width=True, height=300)
                                    
                                    st.info("""
                                    **相关性解读**：
                                    - **🔴 接近1**：强正相关（一科成绩好，另一科成绩也好）
                                    - **🔵 接近-1**：强负相关（一科成绩好，另一科成绩可能差）
                                    - **⚪ 接近0**：无显著相关（两科成绩相对独立）
                                    
                                    **学习建议**：
                                    1. 强正相关的科目可以一起复习
                                    2. 强负相关的科目需要平衡学习时间
                                    3. 无相关的科目可以独立安排学习计划
                                    """)
                                    
                                    # 显示有效数据信息
                                    valid_count = valid_mask.sum()
                                    total_count = len(grades_df)
                                    if valid_count < total_count:
                                        st.warning(f"⚠️ 注意：由于缺失值，分析基于 {valid_count} 个共同考试场次（共 {total_count} 个）")
                            else:
                                st.warning("无法计算相关性矩阵")
                        else:
                            st.warning("没有找到所有科目都有成绩的考试场次，无法进行关联分析")
                else:
                    st.info("需要至少2个科目才能进行关联分析")
            else:
                st.error(f"❌ 未找到学生 {selected_class} - {selected_student} 的成绩数据")


# ============================================
# 辅助函数
# ============================================
def generate_intelligent_suggestions(pattern_info: Dict, prediction_result: Dict, current_grade: float) -> List[str]:
    """
    生成智能学习建议
    
    Args:
        pattern_info: 模式信息
        prediction_result: 预测结果
        current_grade: 当前成绩
        
    Returns:
        学习建议列表
    """
    suggestions = []
    
    # 基于模式
    pattern_type = pattern_info.get("pattern_type", "unknown")
    slope = pattern_info.get("slope", 0)
    trend_strength = pattern_info.get("trend_strength", 0)
    
    if pattern_type in ["strong_upward", "recent_upward"]:
        if trend_strength > 0.5:
            suggestions.append("📈 **强上升趋势**：成绩呈现显著上升趋势，继续保持当前的学习动力和方法。")
        else:
            suggestions.append("↗️ **上升趋势**：成绩有小幅上升，可以考虑适当增加学习强度。")
    
    elif pattern_type in ["strong_downward", "recent_downward"]:
        if trend_strength > 0.5:
            suggestions.append("📉 **强下降趋势**：成绩下降明显，建议立即分析原因并调整学习策略。")
        else:
            suggestions.append("↘️ **轻微下降**：近期成绩有小幅下滑，注意查漏补缺。")
    
    elif pattern_type == "stable":
        suggestions.append("➡️ **成绩稳定**：成绩保持稳定，可以尝试设定更具挑战性的目标。")
    
    elif pattern_type == "volatile":
        suggestions.append("🎢 **成绩波动大**：成绩波动较大，建议分析波动原因，提高稳定性。")
    
    elif "changed_at" in pattern_type:
        change_point = pattern_info.get("change_point", 0)
        suggestions.append(f"🔄 **模式转变**：在第{change_point+1}次考试后成绩模式发生变化，可能由于学习方法或状态改变。")
    
    # 基于预测
    if 'prediction' in prediction_result:
        prediction = prediction_result['prediction']
        confidence = prediction_result.get('confidence_score', 0.5)
        
        improvement = prediction - current_grade
        
        if improvement > 2:
            if confidence > 0.7:
                suggestions.append(f"🔮 **高置信度上升预测**：预计下次考试提高{improvement:.1f}分，置信度{confidence:.1%}。")
            else:
                suggestions.append(f"🔮 **可能上升**：预测成绩可能上升{improvement:.1f}分，但置信度较低({confidence:.1%})。")
        elif improvement < -2:
            if confidence > 0.7:
                suggestions.append(f"⚠️ **高置信度下降预警**：预计下次考试下降{abs(improvement):.1f}分，需要引起重视。")
            else:
                suggestions.append(f"⚠️ **可能下降**：预测成绩可能下降{abs(improvement):.1f}分，建议加强复习。")
        else:
            suggestions.append("➡️ **预测基本稳定**：预计下次考试成绩与当前基本持平。")
    
    # 基于数据量
    data_points = pattern_info.get("data_points", 0)
    if data_points < 4:
        suggestions.append("📊 **数据量较少**：当前只有有限的历史成绩，更多考试数据可以提高分析准确性。")
    
    return suggestions





# ============================================
# 模块5: 班级分析、对比、预测
# ============================================
def module_class_analysis():
    """模块5: 班级分析、对比、预测"""
    st.markdown("## 🏫 5. 班级分析、对比、预测")
    
    if not st.session_state.data_loaded:
        st.warning("请先上传数据文件（切换到'导入数据文件'模块）")
        return
    
    st.markdown("""
    本模块提供班级级别的成绩分析功能，包括：
    1. **班级成绩概览**：查看班级整体成绩情况
    2. **学生排名分析**：分析学生在班级中的排名变化
    3. **班级对比**：对比不同班级的成绩表现
    4. **班级成绩预测**：预测班级整体成绩趋势
    """)
    
    # 选择分析类型
    st.markdown("### 📊 选择分析类型")
    class_analysis_type = st.selectbox(
        "分析类型",
        ["班级成绩概览", "学生排名分析", "班级对比", "班级成绩预测"],
        key="class_analysis_type_select"
    )
    
    if class_analysis_type == "班级成绩概览":
        st.markdown("#### 📊 班级成绩概览")
        st.markdown("查看指定班级的整体成绩情况和学生表现。")
        
        # 选择班级
        classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
        
        selected_class = st.selectbox(
            "选择班级",
            classes,
            key="class_overview_select"
        )
        
        if selected_class:
            st.info(f"正在分析班级: {selected_class}")
            
            # 获取班级学生名单
            class_students = st.session_state.df[
                st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == selected_class
            ][st.session_state.name_column_name].dropna().unique()
            
            st.success(f"✅ 班级 {selected_class} 共有 {len(class_students)} 名学生")
            
            # 显示学生名单
            with st.expander("🏫 查看班级学生名单"):
                for i, student in enumerate(sorted(class_students), 1):
                    st.write(f"{i}. {student}")
            
            # 选择要分析的科目
            score_subjects = [s for s in st.session_state.subjects if '排' not in s]
            
            if score_subjects:
                selected_subject = st.selectbox(
                    "选择要分析的科目",
                    score_subjects,
                    key="class_overview_subject"
                )
                
                if selected_subject:
                    st.markdown(f"#### 📈 {selected_subject} 班级成绩分析")
                    
                    # 收集班级该科目的所有成绩
                    class_grades = []
                    student_names = []
                    
                    for student in class_students[:20]:  # 限制前20名学生，避免性能问题
                        grades_df = GradeManager.get_student_grades(
                            st.session_state.df, selected_class, student,
                            st.session_state.class_column_name, st.session_state.name_column_name,
                            st.session_state.subjects, st.session_state.exams, 
                            st.session_state.column_mapping
                        )
                        
                        if grades_df is not None and selected_subject in grades_df.columns:
                            # 取最近一次考试的成绩
                            latest_grade = pd.to_numeric(grades_df[selected_subject].iloc[-1], errors='coerce')
                            if not pd.isna(latest_grade):
                                class_grades.append(latest_grade)
                                student_names.append(student)
                    
                    if class_grades:
                        # 计算统计指标
                        avg_grade = np.mean(class_grades)
                        max_grade = np.max(class_grades)
                        min_grade = np.min(class_grades)
                        std_grade = np.std(class_grades)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("平均分", f"{avg_grade:.1f}")
                        with col2:
                            st.metric("最高分", f"{max_grade:.1f}")
                        with col3:
                            st.metric("最低分", f"{min_grade:.1f}")
                        with col4:
                            st.metric("标准差", f"{std_grade:.1f}")
                        
                        # 创建成绩分布图
                        st.markdown("##### 📊 成绩分布")
                        
                        fig = go.Figure()
                        
                        # 添加直方图
                        fig.add_trace(go.Histogram(
                            x=class_grades,
                            nbinsx=10,
                            name='成绩分布',
                            marker_color='lightblue',
                            opacity=0.7
                        ))
                        
                        # 添加正态分布曲线
                        x_norm = np.linspace(min_grade, max_grade, 100)
                        y_norm = (1/(std_grade * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - avg_grade) / std_grade) ** 2)
                        y_norm = y_norm * len(class_grades) * (max_grade - min_grade) / 10  # 缩放以匹配直方图
                        
                        fig.add_trace(go.Scatter(
                            x=x_norm,
                            y=y_norm,
                            mode='lines',
                            name='正态分布',
                            line=dict(color='red', width=2)
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_class} {selected_subject}成绩分布",
                            xaxis_title='成绩',
                            yaxis_title='学生人数',
                            height=400,
                            template='plotly_white',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="class_distribution_chart")
                        
                        # 显示成绩排名
                        st.markdown("##### 🏆 成绩排名")
                        
                        # 创建排名表格
                        rank_data = []
                        for i, (grade, student) in enumerate(sorted(zip(class_grades, student_names), reverse=True), 1):
                            rank_data.append({
                                '排名': i,
                                '姓名': student,
                                '成绩': grade,
                                '与平均分差距': grade - avg_grade
                            })
                        
                        rank_df = pd.DataFrame(rank_data)
                        st.dataframe(rank_df, use_container_width=True, height=300)
                        
                        st.info("""
                        **分析说明**：
                        1. 直方图显示成绩分布情况
                        2. 红色曲线为理想的正态分布
                        3. 排名表格显示学生成绩排名
                        """)
                    else:
                        st.warning(f"班级 {selected_class} 中没有找到有效的 {selected_subject} 成绩数据")
            else:
                st.info("没有找到可分析的成绩科目")
    
    elif class_analysis_type == "学生排名分析":
        st.markdown("#### 🏆 学生排名分析")
        st.markdown("分析学生在班级中的排名变化趋势。")
        
        st.info("此功能正在开发中，敬请期待...")
        st.write("功能将包括：")
        st.write("1. 学生排名变化趋势图")
        st.write("2. 排名稳定性分析")
        st.write("3. 排名预测")
    
    elif class_analysis_type == "班级对比":
        st.markdown("#### ⚖️ 班级对比")
        st.markdown("对比不同班级的成绩表现。")
        
        # 选择要对比的班级
        classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
        
        selected_classes = st.multiselect(
            "选择要对比的班级（至少2个）",
            classes,
            default=classes[:min(2, len(classes))],
            key="class_comparison_select"
        )
        
        if len(selected_classes) >= 2:
            st.markdown(f"#### 📊 班级对比: {', '.join(selected_classes)}")
            
            # 选择要对比的科目
            score_subjects = [s for s in st.session_state.subjects if '排' not in s]
            
            if score_subjects:
                selected_subject = st.selectbox(
                    "选择要对比的科目",
                    score_subjects,
                    key="class_comparison_subject"
                )
                
                if selected_subject:
                    st.markdown(f"##### 📈 {selected_subject} 班级对比")
                    
                    # 收集各班级成绩
                    class_stats = []
                    
                    for class_name in selected_classes:
                        # 获取班级学生
                        class_students = st.session_state.df[
                            st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == class_name
                        ][st.session_state.name_column_name].dropna().unique()
                        
                        # 收集成绩
                        class_grades = []
                        for student in class_students[:15]:  # 限制前15名学生
                            grades_df = GradeManager.get_student_grades(
                                st.session_state.df, class_name, student,
                                st.session_state.class_column_name, st.session_state.name_column_name,
                                st.session_state.subjects, st.session_state.exams, 
                                st.session_state.column_mapping
                            )
                            
                            if grades_df is not None and selected_subject in grades_df.columns:
                                latest_grade = pd.to_numeric(grades_df[selected_subject].iloc[-1], errors='coerce')
                                if not pd.isna(latest_grade):
                                    class_grades.append(latest_grade)
                        
                        if class_grades:
                            class_stats.append({
                                '班级': class_name,
                                '平均分': np.mean(class_grades),
                                '最高分': np.max(class_grades),
                                '最低分': np.min(class_grades),
                                '学生数': len(class_grades)
                            })
                    
                    if len(class_stats) >= 2:
                        # 创建对比表格
                        stats_df = pd.DataFrame(class_stats)
                        st.dataframe(stats_df, use_container_width=True)
                        
                        # 创建对比柱状图
                        fig = go.Figure()
                        
                        fig.add_trace(go.Bar(
                            x=[s['班级'] for s in class_stats],
                            y=[s['平均分'] for s in class_stats],
                            name='平均分',
                            marker_color='lightblue',
                            text=[f"{s['平均分']:.1f}" for s in class_stats],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_subject} 班级平均分对比",
                            xaxis_title='班级',
                            yaxis_title='平均分',
                            height=400,
                            template='plotly_white',
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, key="class_comparison_chart")
                        
                        # 创建箱线图对比
                        st.markdown("##### 📦 成绩分布对比")
                        
                        # 收集各班级所有成绩数据
                        box_data = []
                        box_names = []
                        
                        for class_name in selected_classes:
                            class_students = st.session_state.df[
                                st.session_state.df[st.session_state.class_column_name].astype(str).str.strip() == class_name
                            ][st.session_state.name_column_name].dropna().unique()
                            
                            class_grades = []
                            for student in class_students[:15]:
                                grades_df = GradeManager.get_student_grades(
                                    st.session_state.df, class_name, student,
                                    st.session_state.class_column_name, st.session_state.name_column_name,
                                    st.session_state.subjects, st.session_state.exams, 
                                    st.session_state.column_mapping
                                )
                                
                                if grades_df is not None and selected_subject in grades_df.columns:
                                    latest_grade = pd.to_numeric(grades_df[selected_subject].iloc[-1], errors='coerce')
                                    if not pd.isna(latest_grade):
                                        class_grades.append(latest_grade)
                            
                            if class_grades:
                                box_data.append(class_grades)
                                box_names.append(class_name)
                        
                        if len(box_data) >= 2:
                            fig_box = go.Figure()
                            
                            for i, (grades, class_name) in enumerate(zip(box_data, box_names)):
                                fig_box.add_trace(go.Box(
                                    y=grades,
                                    name=class_name,
                                    boxpoints='all',
                                    jitter=0.3,
                                    pointpos=-1.8,
                                    marker_color=f'hsl({(i * 60) % 360}, 70%, 50%)'
                                ))
                            
                            fig_box.update_layout(
                                title=f"{selected_subject} 成绩分布对比",
                                yaxis_title='成绩',
                                height=400,
                                template='plotly_white',
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_box, use_container_width=True, key="class_boxplot")
                            
                            st.info("""
                            **箱线图说明**：
                            1. 箱子中间线为中位数
                            2. 箱子上下边为上下四分位数
                            3. 须线显示数据范围
                            4. 点表示异常值
                            """)
                    else:
                        st.warning("没有足够的数据进行班级对比")
            else:
                st.info("没有找到可对比的成绩科目")
        else:
            st.warning("请至少选择2个班级进行对比")
    
    elif class_analysis_type == "班级成绩预测":
        st.markdown("#### 🔮 班级成绩预测")
        st.markdown("基于班级历史成绩，预测未来考试的整体表现。")
        
        st.info("此功能正在开发中，敬请期待...")
        st.write("功能将包括：")
        st.write("1. 班级平均分预测")
        st.write("2. 班级排名预测")
        st.write("3. 班级进步空间分析")

# ============================================
# 主函数
# ============================================
def main():
    """主应用函数"""
    # 初始化会话状态
    SessionManager.init_session_state()
    
    # 设置中文字体
    FontManager.setup_chinese_font()
    
    # 页面配置
    """
    st.set_page_config(
        page_title="学生成绩查询系统", 
        layout="wide",
        page_icon="🎓"
    )
    """
    # 页面标题
    st.title("🎓 学生成绩查询系统")
    st.markdown("""
    开发者：小基👩‍🌾  
    一个功能完整的学生成绩分析与查询系统
    
    **新增功能**：
    - 🧠 **智能考试权重**：根据考试类型和时间调整权重
    - 📈 **增强趋势分析**：使用加权线性回归进行趋势预测
    - 🔮 **成绩预测**：提供下次考试的成绩预测
    - 💡 **智能学习建议**：基于分析结果提供个性化建议
    """)
    
    # 创建顶边栏
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 1. 导入数据文件",
        "🔍 2. 单个学生查询", 
        "📋 3. 批量学生查询",
        "📈 4. 学生分析预测",
        "🏫 5. 班级分析对比"
    ])
    
    with tab1:
        module_import_data()
    
    with tab2:
        module_single_student_query()
    
    with tab3:
        module_batch_student_query()
    
    with tab4:
        module_student_analysis()
    
    with tab5:
        module_class_analysis()
    
    # 页脚
    st.markdown("---")
    st.caption("© 2026 学生成绩查询系统 | 版本 4.5 | 开发者：小基👩🏻‍🌾 ")

# 运行应用
if __name__ == "__main__":
    main()

