# %%
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    CustomJSFilter,
    CDSView,
    Select,
    IndexFilter,
    GroupFilter,
)
from bokeh.io import show, output_notebook
from bokeh.layouts import column
from bokeh.models.callbacks import CustomJS
from bokeh.models import Button, CustomJS
from bokeh.models import ColorPicker
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.io import curdoc
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.io import output_notebook  # enables plot interface in J notebook
from bokeh.layouts import column, layout, widgetbox, row

# output_notebook()
# %%
# df = pd.DataFrame({'Subject': ['Math', 'Math', 'Math', 'Science', 'Science', 'Science'],
#                   'Class': ['Algebra', 'Calculus', 'Trigonometry', 'Biology', 'Chemistry', 'Physics'],
#                   'FailRate': [0.05, 0.16, 0.31, 0.12, 0.20, 0.08]})
df_melt_slim_super_slim = pd.read_csv("df_melt_slim_super_slim.csv")
source = ColumnDataSource(data=df_melt_slim_super_slim)
callback = CustomJS(code="console.log('tap event occurred')")

index_only = df_melt_slim_super_slim.drop(["iterations", "value"], axis=1)
index_headers = list(index_only.columns)
df_melt_slim_super_slim_indexed = df_melt_slim_super_slim.set_index(index_headers)

select_list = []
for index_header in index_headers:
    index_list = list(index_only[index_header].map(str).unique())
    # try:
    select = Select(title=index_header, value=index_list[0], options=index_list)
    # except:
    #     # index_list.sort()
    #     min_val = min(index_list)
    #     max_val = max(index_list)
    #     select = Slider(title=index_header,start=min_val,end=max_val,
    #                 step=abs(index_list[1]-index_list[0])
    #                 )
    select_list.append(select)
    # slider_list.append(Slider(title="metric", value=1.0, start=0.1, end=5.1))
# df_melt_slim_super_slim_indexed
# view = CDSView(source=src, filters=[GroupFilter(column_name="metric", group="psnr_V")])


p = figure()

scatter = p.scatter(x="iterations", y="value", source=source)
# show(p)

# picker = ColorPicker(title="Line Color")
# picker.js_link('color', scatter.glyph, 'line_color')

# spinner = Spinner(title="Glyph size", low=1, high=40, step=0.5, value=4, width=80)
# spinner.js_link('value', view.glyph, 'size')


# show(column(p, picker))

# group = Slider(title="amplitude", value=1.0, start=-5.0, end=5.0)
# select_0 = select_list[0]
# Set up callbacks
def update(attrname, old, new):
    # Get the current slider values
    group_filter_list = []
    column_name_list = []
    group_list = []
    for selector in select_list:
        column_name = selector._property_values["title"]
        column_name_list.append(column_name)

        group = selector.value
        group_list.append(group)
    print(column_name_list)
    print(group_list)
    data_df = df_melt_slim_super_slim_indexed.xs(group_list, level=column_name_list)
    print(data_df)
    source.data = ColumnDataSource.from_df(data_df)
    # group_filter_list.append(GroupFilter(column_name=column_name, group=group))
    # view = CDSView(source=src, filters=group_filter_list)
    # scatter.view = view


for selector in select_list:
    selector.on_change("value", update)

# inputs
# Set up layout and add to document
inputs = row(select_list)
# inputs.on_change('value', update)
curdoc().add_root(column(inputs, p, width=1200))


# def modify_doc(doc):
#     doc.add_root(column(inputs, p, width=1200))
#     # doc.title = "Sliders"
#     metric.on_change("value", update)


# # %%
# handler = FunctionHandler(modify_doc)
# app = Application(handler)
# # show(app,allow_websocket_origin="*")

# show(app)

# # p.vbar('Class', top='FailRate', width=0.9, source=src, view=view)

# subj_list = sorted(list(set(src.data['psf_scale'])))

# callback = CustomJS(args=dict(src=src), code='''
#     src.change.emit();
# ''')

# js_filter = CustomJSFilter(code='''
# var indices = [];
# for (var i = 0; i < src.get_length(); i++){
#     if (src.data['Subject'][i] == select.value){
#         indices.push(true);
#     } else {
#         indices.push(false);
#     }
# }
# return indices;
# ''')

# options = ['Please select...'] + subj_list
# select = Select(title='Subject Selection', value=options[0], options=options)

# select.js_on_change('value', callback)

# filter = IndexFilter(indices=[])

# callback = CustomJS(args=dict(src=src, filter=filter), code='''
#   const indices = []
#   for (var i = 0; i < src.get_length(); i++) {
#     console.log(i, src.data['Subject'][i], cb_obj.value)
#     if (src.data['Subject'][i] == cb_obj.value) {
#       indices.push(i)
#     }
#   }
#   filter.indices = indices
#   src.change.emit()
# ''')

# select.js_on_change('value', callback)

# view = CDSView(source=src, filters=[filter])
# class_list = sorted(list(src.data['Class']))

# p = figure(x_range=class_list, plot_height=400, plot_width=400)
# p.vbar('Class', top='FailRate', width=0.9, source=src, view=view)

# show(column(select, p))


# %%
