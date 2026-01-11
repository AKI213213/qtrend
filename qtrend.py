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
            '期末': 1.0, '期中': 0.85, '月考': 0.7, '模拟': 0.8,
            '联考': 0.9, '五校': 0.88, '八校': 0.86, '质检': 0.75
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
    1. 在左侧选择班级和学生
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
# 模块4: 学生成绩分析、预测（修复数组形状错误）
# ============================================
def module_student_analysis():
    """模块4: 学生成绩分析、预测（增强版）"""
    st.markdown("## 📈 4. 学生成绩分析、预测")
    
    if not st.session_state.data_loaded:
        st.warning("请先上传数据文件（切换到'导入数据文件'模块）")
        return
    
    st.markdown("""
    **增强版功能特色**：
    1. **智能考试权重计算**：根据考试类型（期末、期中、月考等）和时间衰减自动调整权重
    2. **加权线性回归预测**：考虑不同考试的重要性，提供更准确的趋势预测
    3. **置信区间分析**：提供预测结果的置信区间，评估预测可靠性
    4. **成绩异常检测**：识别异常波动的成绩，帮助发现问题
    5. **科目关联分析**：分析科目之间的相关性，发现学习模式
    """)
    
    # 选择分析类型
    st.markdown("### 📊 选择分析类型")
    analysis_type = st.selectbox(
        "分析类型",
        ["成绩趋势分析（增强版）", "成绩预测（增强版）", "成绩异常检测", "科目关联分析"],
        key="analysis_type_select"
    )
    
    if "增强版" in analysis_type:
        if "趋势分析" in analysis_type:
            st.markdown("#### 📈 增强版成绩趋势分析")
            st.markdown("选择要分析的学生和科目，系统将提供详细的趋势分析报告，包含考试权重分析和趋势预测。")
        else:
            st.markdown("#### 🔮 增强版成绩预测")
            st.markdown("基于学生历史成绩，使用增强算法预测未来考试的成绩，包含置信区间和趋势分析。")
        
        # 选择班级和学生
        classes = sorted(st.session_state.df[st.session_state.class_column_name].dropna().astype(str).str.strip().unique())
        
        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox(
                "选择班级",
                classes,
                key="analysis_class_select"
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
                        key="analysis_student_select"
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
                        key="enhanced_analysis_subject"
                    )
                    
                    if selected_subject and selected_subject in grades_df.columns:
                        # 修复：确保成绩数据和考试场次数据长度一致
                        # 提取成绩数据和考试场次
                        subject_data = pd.to_numeric(grades_df[selected_subject], errors='coerce')
                        
                        # 创建一个布尔掩码，标识非空值
                        valid_mask = subject_data.notna()
                        
                        # 获取有效成绩
                        valid_data = subject_data[valid_mask]
                        
                        # 获取对应的考试场次
                        exam_names = grades_df.loc[valid_mask, '考试场次'].tolist()
                        
                        if len(valid_data) >= 3:  # 至少需要3个数据点
                            st.markdown("#### 📊 高级分析报告")
                            
                            # 检查数据一致性
                            if len(valid_data) != len(exam_names):
                                st.error(f"数据不一致：成绩数量({len(valid_data)}) ≠ 考试场次数量({len(exam_names)})")
                                st.warning("可能存在数据质量问题，请检查原始数据")
                                return
                            
                            # 创建分析器
                            enhanced_analyzer = EnhancedGradeTrendAnalyzer()
                            
                            # 进行分析
                            with st.spinner("正在进行深度分析..."):
                                # 获取考试权重
                                exam_weights = []
                                for i, exam_name in enumerate(exam_names):
                                    is_recent = (i >= len(exam_names) - 3)
                                    weight = enhanced_analyzer.weight_calculator.calculate_exam_weight(
                                        exam_name, i, len(exam_names), is_recent
                                    )
                                    exam_weights.append(weight)
                                
                                # 增强趋势分析
                                trend_result = enhanced_analyzer.calculate_trend_stats(
                                    valid_data.values, exam_names
                                )
                            
                            # 显示考试权重
                            st.markdown("##### 📊 考试权重分析")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("最近考试权重", f"{exam_weights[-1]:.2f}", 
                                         f"权重最高: {max(exam_weights):.2f}")
                            with col2:
                                st.metric("权重波动", f"{np.std(exam_weights):.3f}",
                                         f"平均权重: {np.mean(exam_weights):.2f}")
                            with col3:
                                recent_exams = [exam for i, exam in enumerate(exam_names) 
                                              if i >= len(exam_names) - 3]
                                st.metric("近期考试数", len(recent_exams), 
                                         f"{', '.join(recent_exams)}")
                            
                            # 显示趋势分析结果
                            st.markdown("##### 📈 趋势分析")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                trend_emoji = "📈" if trend_result['slope'] > 0 else "📉" if trend_result['slope'] < 0 else "➡️"
                                st.metric("成绩趋势", f"{trend_result['trend']} {trend_emoji}",
                                         f"斜率: {trend_result['slope']:.3f}")
                            with col2:
                                stability_level = "高" if trend_result['stability'] < 0.1 else "中" if trend_result['stability'] < 0.2 else "低"
                                st.metric("成绩稳定性", stability_level,
                                         f"标准差/均值: {trend_result['stability']:.3f}")
                            with col3:
                                st.metric("历史平均分", f"{trend_result['mean_grade']:.1f}",
                                         f"最新成绩: {trend_result['current_grade']:.1f}")
                            
                            # 显示预测结果
                            st.markdown("##### 🔮 预测结果")
                            if 'next_grade' in trend_result:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("下次考试预测", f"{trend_result['next_grade']:.1f}",
                                             f"基于加权线性回归")
                                with col2:
                                    confidence_margin = 5 + 2 * (1 - trend_result.get('stability', 0.5))
                                    st.metric("预测可靠性", f"±{confidence_margin:.1f}",
                                             f"95%置信区间")
                                with col3:
                                    if trend_result['next_grade'] > trend_result['current_grade']:
                                        improvement = trend_result['next_grade'] - trend_result['current_grade']
                                        st.metric("预计提升", f"+{improvement:.1f}",
                                                 f"与当前成绩相比")
                            else:
                                st.info("无法进行预测，需要更多数据")
                            
                            # 创建增强版趋势图表
                            st.markdown("##### 📈 可视化分析")
                            
                            # 创建一个过滤后的DataFrame，只包含有效成绩
                            valid_grades_df = grades_df.loc[valid_mask].copy()
                            
                            # 创建趋势图表
                            fig = ChartGenerator.create_grade_trend_chart(
                                valid_grades_df, [selected_subject], selected_student, selected_class
                            )
                            
                            if fig:
                                # 添加趋势线
                                x_vals = list(range(len(valid_grades_df['考试场次'])))
                                y_vals = trend_result['intercept'] + trend_result['slope'] * np.array(x_vals)
                                
                                # 添加预测点
                                if 'next_grade' in trend_result:
                                    x_pred = [len(valid_grades_df['考试场次'])]
                                    y_pred = [trend_result['next_grade']]
                                    
                                    # 创建趋势线
                                    fig.add_trace(go.Scatter(
                                        x=valid_grades_df['考试场次'],
                                        y=y_vals,
                                        mode='lines',
                                        name='趋势线',
                                        line=dict(color='red', width=2, dash='dash')
                                    ))
                                    
                                    # 创建预测点
                                    fig.add_trace(go.Scatter(
                                        x=[f'预测{len(valid_grades_df)+1}'],
                                        y=y_pred,
                                        mode='markers',
                                        name='预测成绩',
                                        marker=dict(size=12, color='green', symbol='star')
                                    ))
                                
                                st.plotly_chart(fig, use_container_width=True, key="enhanced_trend_chart")
                            
                            # 显示缺失数据信息
                            missing_count = len(subject_data) - len(valid_data)
                            if missing_count > 0:
                                st.warning(f"⚠️ 注意：该科目有 {missing_count} 个缺失值，已自动过滤。分析基于 {len(valid_data)} 个有效成绩。")
                            
                            # 显示学习建议
                            st.markdown("##### 💡 学习建议")
                            suggestions = generate_enhanced_suggestions(trend_result)
                            for suggestion in suggestions:
                                st.info(suggestion)
                            
                            # 数据下载
                            st.markdown("##### 💾 下载分析报告")
                            analysis_report = {
                                "学生信息": f"{selected_class} - {selected_student}",
                                "科目": selected_subject,
                                "分析时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "有效成绩数": len(valid_data),
                                "缺失成绩数": missing_count,
                                "趋势分析": trend_result,
                                "考试权重": exam_weights
                            }
                            
                            # 转换为DataFrame
                            report_df = pd.DataFrame([analysis_report])
                            report_csv = report_df.to_csv(index=False, encoding='utf-8-sig')
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📥 下载分析报告(CSV)",
                                    data=report_csv,
                                    file_name=f"{selected_class}_{selected_student}_{selected_subject}_分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    key="enhanced_analysis_csv"
                                )
                            
                            with col2:
                                # 生成分析摘要
                                summary_text = generate_analysis_summary(selected_student, selected_subject, trend_result, missing_count)
                                st.download_button(
                                    label="📝 下载分析摘要(TXT)",
                                    data=summary_text,
                                    file_name=f"{selected_class}_{selected_student}_{selected_subject}_分析摘要_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain",
                                    key="enhanced_analysis_txt"
                                )
                            
                        else:
                            st.warning(f"需要至少3次有效成绩才能进行深度分析，当前只有{len(valid_data)}次")
                            if len(subject_data) > len(valid_data):
                                missing_count = len(subject_data) - len(valid_data)
                                st.info(f"该科目共有 {len(subject_data)} 次考试，其中 {missing_count} 次考试无成绩")
                else:
                    st.info("没有找到可分析的成绩科目")
            else:
                st.error(f"❌ 未找到学生 {selected_class} - {selected_student} 的成绩数据")
    
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
                
                # 选择要分析的科目
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
                                
                                # 修复：确保使用有效成绩
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
                                    
                                    # 修复：确保使用有效成绩
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
                        
                        # 修复：确保所有科目在相同考试场次上都有成绩
                        # 获取所有科目都有成绩的考试场次
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
                                
                                # 只显示增强版热力图
                                st.markdown("##### 📈 科目相关性热力图")
                                
                                # 创建增强版热力图 - 修复字体颜色问题
                                correlation_values = correlation_matrix.values
                                
                                # 创建热力图（不显示文本）
                                fig = go.Figure(data=go.Heatmap(
                                    z=correlation_values,
                                    x=correlation_matrix.columns,
                                    y=correlation_matrix.index,
                                    colorscale='RdBu',  # 使用红蓝渐变
                                    zmin=-1,
                                    zmax=1,
                                    hoverongaps=False,
                                    hoverinfo='x+y+z',
                                    zmid=0,
                                ))
                                
                                # 为每个单元格添加annotation，可以单独设置字体颜色
                                annotations = []
                                for i in range(len(correlation_matrix.index)):
                                    for j in range(len(correlation_matrix.columns)):
                                        value = correlation_values[i, j]
                                        
                                        # 根据相关性绝对值决定字体颜色
                                        if abs(value) > 0.5:
                                            font_color = 'white'  # 深色背景用白色字体
                                        else:
                                            font_color = 'black'  # 浅色背景用黑色字体
                                        
                                        # 创建annotation
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
                                    annotations=annotations  # 添加所有annotations
                                )
                                
                                # 设置热力图的样式
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


def generate_enhanced_suggestions(trend_result):
    """生成增强版学习建议"""
    suggestions = []
    
    # 基于趋势分析
    if trend_result.get('trend') == "上升趋势":
        suggestions.append("📈 **成绩呈显著上升趋势**：继续保持当前的学习方法和节奏，巩固优势科目的学习效果。")
    elif trend_result.get('trend') == "轻微上升":
        suggestions.append("↗️ **成绩有小幅提升**：在保持现有学习方法的基础上，可以尝试加强薄弱环节的练习。")
    elif trend_result.get('trend') == "下降趋势":
        suggestions.append("📉 **成绩呈下降趋势**：建议分析最近的学习状态，找出成绩下降的原因并及时调整学习策略。")
    elif trend_result.get('trend') == "轻微下降":
        suggestions.append("↘️ **成绩有小幅下滑**：注意近期考试的失分点，加强相关知识点的复习。")
    else:
        suggestions.append("➡️ **成绩保持平稳**：可以尝试设定更具挑战性的学习目标，推动成绩进一步提升。")
    
    # 基于稳定性
    stability = trend_result.get('stability', 0)
    if stability < 0.1:
        suggestions.append("🎯 **成绩非常稳定**：说明学习状态稳定，可以尝试挑战更高难度的内容。")
    elif stability > 0.3:
        suggestions.append("🎢 **成绩波动较大**：建议分析波动原因，可能是知识点掌握不牢固或考试状态不稳定。")
    
    # 基于预测结果
    if 'next_grade' in trend_result:
        if trend_result['next_grade'] > trend_result.get('current_grade', 0):
            suggestions.append(f"🔮 **预测成绩上升**：预计下次考试成绩为{trend_result['next_grade']:.1f}分，比当前提高{(trend_result['next_grade']-trend_result.get('current_grade', 0)):.1f}分。")
        elif trend_result['next_grade'] < trend_result.get('current_grade', 0):
            suggestions.append(f"⚠️ **预测成绩下降**：预计下次考试成绩为{trend_result['next_grade']:.1f}分，比当前下降{(trend_result.get('current_grade', 0)-trend_result['next_grade']):.1f}分，需要加强复习。")
    
    return suggestions


def generate_analysis_summary(student_name, subject, trend_result, missing_count=0):
    """生成分析摘要"""
    summary = f"""
========== 学生成绩分析报告 ==========

学生姓名：{student_name}
分析科目：{subject}
分析时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
缺失成绩数：{missing_count}

【趋势分析结果】
当前成绩：{trend_result.get('current_grade', 0):.1f}
平均成绩：{trend_result.get('mean_grade', 0):.1f}
成绩趋势：{trend_result.get('trend', '未知')}
趋势强度：{trend_result.get('slope', 0):.3f}
稳定性指标：{trend_result.get('stability', 0):.3f}

【预测结果】
下次考试预测：{trend_result.get('next_grade', 0):.1f}

【学习建议】
"""
    
    suggestions = generate_enhanced_suggestions(trend_result)
    for i, suggestion in enumerate(suggestions, 1):
        summary += f"{i}. {suggestion}\n"
    
    summary += """
======================================
备注：本报告基于加权线性回归算法生成，仅供参考。
"""
    
    return summary




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
    st.set_page_config(
        page_title="学生成绩查询系统 - 增强版", 
        layout="wide",
        page_icon="🎓"
    )
    
    # 页面标题
    st.title("🎓 学生成绩查询系统 - 增强版")
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
        "📈 4. 学生分析预测（增强版）",
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
    st.caption("© 2026 学生成绩查询系统 - 增强版 | 版本 3.0 | 开发者：小基👩🏻‍🌾 ")

# 运行应用
if __name__ == "__main__":
    main()
