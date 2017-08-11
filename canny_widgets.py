import numpy as np
from ipywidgets import widgets, Layout
import os
from pint import UnitRegistry

ur = UnitRegistry()


def filename_widget(description):
    entry = widgets.Text(
        value='', placeholder='filename', description=description,
        disabled=False)
    validation = widgets.Valid(value=False, description='file exists')

    def observer(change):
        try:
            filename = change['new']['value']
            validation.value = os.path.exists(filename) \
                and os.path.isfile(filename)

        except (KeyError, TypeError):
            pass
    
    entry.observe(observer)
    return widgets.Box([entry, validation]), entry


def quantity_widget(description, dimension):
    entry = widgets.Text(value='', placeholder='quantity of {}'.format(dimension),
                         description=description, disabled=False)
    validation = widgets.Valid(value=False, description='correct units')

    def observer(change):
        try:
            s = change['new']['value']
            v = ur(s)
            validation.value = (v.dimensionality == dimension)
            
        except Exception:
            pass
    
    entry.observe(observer)
    return widgets.Box([entry, validation]), entry


def make_input_box(config):
    input_file_box, input_file_text = filename_widget('Input file: ')
    control_file_box, control_file_text = filename_widget('Control sample: ')

    longitude_kw = widgets.Text(value='lon', description='Longitude keyword: ')
    lattitude_kw = widgets.Text(value='lat', description='Lattitude keyword: ')
    value_kw = widgets.Text(
        value='', placeholder='value name', description='Value keyword: ')

    load_button = widgets.HBox(
        [widgets.Button(description='Load', button_style='info')],
        layout=Layout(justify_content='flex-end'))

    input_box = widgets.VBox([
        input_file_box, control_file_box, longitude_kw,
        lattitude_kw, value_kw, load_button])

    return input_box


def make_selection_box(config):
    tab_contents = ['All', 'Monthly']
    children = [widgets.Label(value=name) for name in tab_contents]
    tab = widgets.Tab()

    month = widgets.Dropdown(
        options=dict(zip(
            ['January', 'February', 'March', 'April', 'May', 'June'
             'July', 'August', 'September', 'October', 'November', 'December'],
            range(12))),
        value=0,
        description='Month:',
        disabled=False,
    )
    children[1] = month

    tab.children = children
    for i in range(len(children)):
        tab.set_title(i, tab_contents[i])

    selection_box = widgets.HBox([tab])

    config.expose("selection_tab", tab)
    config.expose("month", month)
    return selection_box


def make_filtering_box(config):
    sigma_l_box, sigma_l = quantity_widget(
        "spatial smoothing radius (gaussian std-dev)", "[length]")
    sigma_t_box, sigma_t = quantity_widget(
        "time smoothing interval (gaussian std-dev)", "[time]")
    sensity_box, sensitivity_ratio = quantity_widget(
        "sensitivity ratio", "[length]/[time]")

    filtering_box = widgets.VBox([sigma_l_box, sigma_t_box, sensity_box])
    config.expose("sigma_l", sigma_l)
    config.expose("sigma_t", sigma_t)
    config.expose("sensitivity_ratio", sensitivity_ratio)
    return filtering_box


class ConfigWidget(widgets.Accordion):
    def __init__(self):
        self.exposed_widgets = {}
        super(ConfigWidget, self).__init__(children=[
            make_input_box(self),
            make_selection_box(self),
            filtering_box(self)])
        self.set_title(0, 'Input specification')
        self.set_title(1, 'Selection')
        self.set_title(2, 'Filtering')

    def expose(self, name, widget):
        self.exposed_widgets[name] = widget

    def __getitem__(self, item):
        return self.exposed_widgets[item]

