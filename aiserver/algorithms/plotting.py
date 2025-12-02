from .base import Algorithm, AlgorithmParameter, Port

# 箱型图绘制
box_plot = Algorithm(
    id="box_plot",
    name="箱型图绘制",
    category="数据绘图",
    prompt="请对 {VAR_NAME} 绘制箱型图，展示数据分布特征。支持单变量、多变量和分组箱型图。",
    parameters=[
        AlgorithmParameter(name="y_columns", type="list", default=[], label="Y轴列名", description="要绘制箱型图的列 (留空则绘制所有数值列)", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="group_by", type="str", default="", label="分组列", description="用于分组的列名 (可选)", widget="column-selector", priority="non-critical"),
        AlgorithmParameter(name="layout", type="str", default="子图", label="排布方式", options=["子图", "一张图"], description="选择图表的排布方式：子图（每行4个）或所有数据显示在一张图", priority="non-critical"),
        AlgorithmParameter(name="title", type="str", default="箱型图", label="图表标题", description="图表的标题", priority="non-critical"),
        AlgorithmParameter(name="xlabel", type="str", default="", label="X轴标签", description="X轴的显示标签", priority="non-critical"),
        AlgorithmParameter(name="ylabel", type="str", default="", label="Y轴标签", description="Y轴的显示标签", priority="non-critical"),
        AlgorithmParameter(name="show_outliers", type="bool", default=True, label="显示异常值", description="是否在箱型图中显示异常值", priority="non-critical"),
        AlgorithmParameter(name="figsize", type="str", default="(12, 6)", label="图像尺寸", description="图像大小元组，例如 (12, 6)", priority="non-critical")
    ],
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns"],
    inputs=[Port(name="df_in")],
    outputs=[],  # No output node
    template="""
# Box Plot for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# Get parameters
y_columns = {y_columns}
group_by = '{group_by}'
layout = '{layout}'
title = '{title}'
xlabel = '{xlabel}'
ylabel = '{ylabel}'
show_outliers = {show_outliers}
figsize_str = '{figsize}'

# Parse figsize
try:
    figsize = eval(figsize_str)
except:
    figsize = (12, 6)

# Determine Y columns
if not y_columns:
    # Use all numeric columns if none specified
    y_columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

# Filter to selected columns
plot_data = {OUTPUT_VAR}[y_columns].copy()

# Check if we have a group_by column
if group_by and group_by in {OUTPUT_VAR}.columns:
    # Add group column to plot_data
    plot_data[group_by] = {OUTPUT_VAR}[group_by]
    
    # Create grouped box plot
    
    # If only one Y column, use seaborn for better grouping
    if len(y_columns) == 1:
        plt.figure(figsize=figsize)
        sns.boxplot(x=group_by, y=y_columns[0], data=plot_data, showfliers=show_outliers)
        plt.title(f"Box Plot of {y_columns[0]} by {group_by}")
    else:
        # For multiple Y columns, use subplots based on layout
        if layout == "一张图":
            # Multiple Y columns, single plot with different colors
            plt.figure(figsize=figsize)
            for i, col in enumerate(y_columns):
                sns.boxplot(x=group_by, y=col, data=plot_data, showfliers=show_outliers, alpha=0.7, label=col)
            plt.title(f"Box Plots of Multiple Columns by {group_by}")
            plt.legend()
        else:
            # Multiple Y columns - grid subplots (4 per row)
            n_cols = 4
            n_rows = (len(y_columns) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols/2, figsize[1]*n_rows))
            axes = axes.flatten()
            
            for i, col in enumerate(y_columns):
                sns.boxplot(x=group_by, y=col, data=plot_data, ax=axes[i], showfliers=show_outliers)
                axes[i].set_title(f"Box Plot of {col} by {group_by}")
                axes[i].tick_params(axis='x', rotation=45)
            
            # Hide unused axes
            for i in range(len(y_columns), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
else:
    # Create regular box plot
    if len(y_columns) == 1:
        # Single column - simple box plot
        plt.figure(figsize=figsize)
        plt.boxplot(plot_data[y_columns[0]].dropna(), showfliers=show_outliers)
        plt.title(f"Box Plot of {y_columns[0]}")
        plt.xticks([1], y_columns)
    else:
        if layout == "一张图":
            # Multiple columns - single plot with overlay
            plt.figure(figsize=figsize)
            for col in y_columns:
                sns.boxplot(data=plot_data[col].dropna(), alpha=0.5, showfliers=show_outliers, label=col)
            plt.title("Box Plots of Multiple Columns")
            plt.legend()
            plt.xticks(rotation=45)
        else:
            # Multiple columns - grid subplots (4 per row)
            n_cols = 4
            n_rows = (len(y_columns) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols/2, figsize[1]*n_rows))
            axes = axes.flatten()
            
            for i, col in enumerate(y_columns):
                plt.boxplot(plot_data[col].dropna(), showfliers=show_outliers, ax=axes[i])
                axes[i].set_title(f"Box Plot of {col}")
                axes[i].set_xticklabels([col])
            
            # Hide unused axes
            for i in range(len(y_columns), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()

# Set labels if provided
if xlabel:
    plt.xlabel(xlabel)
if ylabel:
    plt.ylabel(ylabel)

# Final adjustments
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""
)

# 直方图显示
histogram = Algorithm(
    id="histogram",
    name="直方图显示",
    category="数据绘图",
    prompt="请对 {VAR_NAME} 绘制直方图，展示数据分布特征。支持自定义箱数、颜色、密度曲线等设置。",
    parameters=[
        AlgorithmParameter(name="y_columns", type="list", default=[], label="Y轴列名", description="要绘制直方图的列 (留空则绘制所有数值列)", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="bins", type="int", default=30, label="箱数", description="直方图的箱数", min=5, max=100, priority="non-critical"),
        AlgorithmParameter(name="kde", type="bool", default=True, label="显示密度曲线", description="是否在直方图上显示密度曲线", priority="non-critical"),
        AlgorithmParameter(name="layout", type="str", default="子图", label="排布方式", options=["子图", "一张图"], description="选择图表的排布方式：子图（每行4个）或所有数据显示在一张图", priority="non-critical"),
        AlgorithmParameter(name="title", type="str", default="直方图", label="图表标题", description="直方图的标题", priority="non-critical"),
        AlgorithmParameter(name="xlabel", type="str", default="", label="X轴标签", description="X轴的显示标签", priority="non-critical"),
        AlgorithmParameter(name="ylabel", type="str", default="", label="Y轴标签", description="Y轴的显示标签", priority="non-critical"),
        AlgorithmParameter(name="figsize", type="str", default="(12, 6)", label="图像尺寸", description="图像大小元组，例如 (12, 6)", priority="non-critical")
    ],
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns"],
    inputs=[Port(name="df_in")],
    outputs=[],  # No output node
    template="""
# Histogram for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# Get parameters
y_columns = {y_columns}
bins = {bins}
kde = {kde}
layout = '{layout}'
title = '{title}'
xlabel = '{xlabel}'
ylabel = '{ylabel}'
figsize_str = '{figsize}'

# Parse figsize
try:
    figsize = eval(figsize_str)
except:
    figsize = (12, 6)

# Determine Y columns
if not y_columns:
    # Use all numeric columns if none specified
    y_columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

# Filter to selected columns
plot_data = {OUTPUT_VAR}[y_columns].copy()

# Create histogram
if len(y_columns) == 1:
    # Single column - simple histogram
    plt.figure(figsize=figsize)
    sns.histplot(plot_data[y_columns[0]].dropna(), bins=bins, kde=kde)
    plt.title(f"Histogram of {y_columns[0]}")
    plt.xticks(rotation=45)
else:
    if layout == "一张图":
        # Multiple columns - single plot with overlay
        plt.figure(figsize=figsize)
        for col in y_columns:
            sns.histplot(plot_data[col].dropna(), bins=bins, kde=kde, alpha=0.5, label=col)
        plt.title("Histogram of Multiple Columns")
        plt.legend()
        plt.xticks(rotation=45)
    else:
        # Multiple columns - grid subplots (4 per row)
        n_cols = 4
        n_rows = (len(y_columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols/2, figsize[1]*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(y_columns):
            sns.histplot(plot_data[col].dropna(), bins=bins, kde=kde, ax=axes[i])
            axes[i].set_title(f"Histogram of {col}")
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused axes
        for i in range(len(y_columns), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()

# Set labels if provided
if xlabel:
    plt.xlabel(xlabel)
if ylabel:
    plt.ylabel(ylabel)

plt.tight_layout()
plt.show()
"""
)

# 密度图显示
density_plot = Algorithm(
    id="density_plot",
    name="密度图显示",
    category="数据绘图",
    prompt="请对 {VAR_NAME} 绘制密度图，展示数据分布特征。支持自定义颜色、带宽等设置。",
    parameters=[
        AlgorithmParameter(name="y_columns", type="list", default=[], label="Y轴列名", description="要绘制密度图的列 (留空则绘制所有数值列)", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="layout", type="str", default="子图", label="排布方式", options=["子图", "一张图"], description="选择图表的排布方式：子图（每行4个）或所有数据显示在一张图", priority="non-critical"),
        AlgorithmParameter(name="title", type="str", default="密度图", label="图表标题", description="密度图的标题", priority="non-critical"),
        AlgorithmParameter(name="xlabel", type="str", default="", label="X轴标签", description="X轴的显示标签", priority="non-critical"),
        AlgorithmParameter(name="ylabel", type="str", default="", label="Y轴标签", description="Y轴的显示标签", priority="non-critical"),
        AlgorithmParameter(name="figsize", type="str", default="(12, 6)", label="图像尺寸", description="图像大小元组，例如 (12, 6)", priority="non-critical")
    ],
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns"],
    inputs=[Port(name="df_in")],
    outputs=[],  # No output node
    template="""
# Density Plot for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# Get parameters
y_columns = {y_columns}
layout = '{layout}'
title = '{title}'
xlabel = '{xlabel}'
ylabel = '{ylabel}'
figsize_str = '{figsize}'

# Parse figsize
try:
    figsize = eval(figsize_str)
except:
    figsize = (12, 6)

# Determine Y columns
if not y_columns:
    # Use all numeric columns if none specified
    y_columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

# Filter to selected columns
plot_data = {OUTPUT_VAR}[y_columns].copy()

# Create density plot
if len(y_columns) == 1:
    # Single column - simple density plot
    plt.figure(figsize=figsize)
    sns.kdeplot(plot_data[y_columns[0]].dropna(), fill=True)
    plt.title(f"Density Plot of {y_columns[0]}")
    plt.xticks(rotation=45)
else:
    if layout == "一张图":
        # Multiple columns - single plot with multiple lines
        plt.figure(figsize=figsize)
        for col in y_columns:
            sns.kdeplot(plot_data[col].dropna(), fill=True, alpha=0.5, label=col)
        plt.title("Density Plot of Multiple Columns")
        plt.legend()
        plt.xticks(rotation=45)
    else:
        # Multiple columns - grid subplots (4 per row)
        n_cols = 4
        n_rows = (len(y_columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols/2, figsize[1]*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(y_columns):
            sns.kdeplot(plot_data[col].dropna(), fill=True, ax=axes[i])
            axes[i].set_title(f"Density Plot of {col}")
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused axes
        for i in range(len(y_columns), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()

# Set labels if provided
if xlabel:
    plt.xlabel(xlabel)
if ylabel:
    plt.ylabel(ylabel)

plt.tight_layout()
plt.show()
"""
)

# 小提琴图显示
violin_plot = Algorithm(
    id="violin_plot",
    name="小提琴图显示",
    category="数据绘图",
    prompt="请对 {VAR_NAME} 绘制小提琴图，展示数据分布特征。支持自定义颜色、带宽等设置。",
    parameters=[
        AlgorithmParameter(name="y_columns", type="list", default=[], label="Y轴列名", description="要绘制小提琴图的列 (留空则绘制所有数值列)", widget="column-selector", priority="critical"),
        AlgorithmParameter(name="layout", type="str", default="子图", label="排布方式", options=["子图", "一张图"], description="选择图表的排布方式：子图（每行4个）或所有数据显示在一张图", priority="non-critical"),
        AlgorithmParameter(name="title", type="str", default="小提琴图", label="图表标题", description="小提琴图的标题", priority="non-critical"),
        AlgorithmParameter(name="xlabel", type="str", default="", label="X轴标签", description="X轴的显示标签", priority="non-critical"),
        AlgorithmParameter(name="ylabel", type="str", default="", label="Y轴标签", description="Y轴的显示标签", priority="non-critical"),
        AlgorithmParameter(name="figsize", type="str", default="(12, 6)", label="图像尺寸", description="图像大小元组，例如 (12, 6)", priority="non-critical")
    ],
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns"],
    inputs=[Port(name="df_in")],
    outputs=[],  # No output node
    template="""
# Violin Plot for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# Get parameters
y_columns = {y_columns}
layout = '{layout}'
title = '{title}'
xlabel = '{xlabel}'
ylabel = '{ylabel}'
figsize_str = '{figsize}'

# Parse figsize
try:
    figsize = eval(figsize_str)
except:
    figsize = (12, 6)

# Determine Y columns
if not y_columns:
    # Use all numeric columns if none specified
    y_columns = {OUTPUT_VAR}.select_dtypes(include=['number']).columns.tolist()

# Filter to selected columns
plot_data = {OUTPUT_VAR}[y_columns].copy()

# Create violin plot
if len(y_columns) == 1:
    # Single column - simple violin plot
    plt.figure(figsize=figsize)
    sns.violinplot(data=plot_data[y_columns[0]].dropna())
    plt.title(f"Violin Plot of {y_columns[0]}")
    plt.xticks([0], y_columns)
else:
    if layout == "一张图":
        # Multiple columns - single plot with overlay
        plt.figure(figsize=figsize)
        for col in y_columns:
            sns.violinplot(data=plot_data[col].dropna(), alpha=0.5, label=col)
        plt.title("Violin Plot of Multiple Columns")
        plt.legend()
        plt.xticks(rotation=45)
    else:
        # Multiple columns - grid subplots (4 per row)
        n_cols = 4
        n_rows = (len(y_columns) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0]*n_cols/2, figsize[1]*n_rows))
        axes = axes.flatten()
        
        for i, col in enumerate(y_columns):
            sns.violinplot(data=plot_data[col].dropna(), ax=axes[i])
            axes[i].set_title(f"Violin Plot of {col}")
            axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused axes
        for i in range(len(y_columns), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()

# Set labels if provided
if xlabel:
    plt.xlabel(xlabel)
if ylabel:
    plt.ylabel(ylabel)

plt.tight_layout()
plt.show()
"""
)

# 相关性热力图
correlation_heatmap = Algorithm(
    id="correlation_heatmap",
    name="相关性热力图",
    category="数据绘图",
    prompt="请对 {VAR_NAME} 进行相关性分析。使用seaborn的heatmap函数，绘制各列之间的相关性热力图。",
    parameters=[
        AlgorithmParameter(name="method", type="str", default="pearson", label="相关系数方法", options=["pearson", "kendall", "spearman"], description="计算相关系数的方法", priority="non-critical"),
        AlgorithmParameter(name="title", type="str", default="相关性热力图", label="图表标题", description="图表的标题", priority="non-critical"),
        AlgorithmParameter(name="figsize", type="str", default="(12, 10)", label="图像尺寸", description="图像大小元组，例如 (12, 10)", priority="non-critical"),
        AlgorithmParameter(name="annot", type="bool", default=True, label="显示数值", description="是否在热力图上显示相关系数数值", priority="non-critical"),
        AlgorithmParameter(name="cmap", type="str", default="coolwarm", label="颜色映射", description="热力图的颜色映射方案", priority="non-critical")
    ],
    imports=["import pandas as pd", "import matplotlib.pyplot as plt", "import seaborn as sns"],
    inputs=[Port(name="df_in")],
    outputs=[],  # No output node
    template="""
# Correlation Heatmap for {VAR_NAME}
{OUTPUT_VAR} = {VAR_NAME}.copy()

# Get parameters
method = '{method}'
title = '{title}'
figsize_str = '{figsize}'
annot = {annot}
cmap = '{cmap}'

# Parse figsize
try:
    figsize = eval(figsize_str)
except:
    figsize = (12, 10)

# Calculate correlation matrix
numeric_data = {OUTPUT_VAR}.select_dtypes(include=['number'])
corr_matrix = numeric_data.corr(method=method)

# Create heatmap
plt.figure(figsize=figsize)
sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt='.2f', linewidths=0.5)
plt.title(title)
plt.tight_layout()
plt.show()
"""
)

algorithms = [box_plot, histogram, density_plot, violin_plot, correlation_heatmap]
