import pandas as pd
from classes.hardware.architecture.dimension import Dimension


# df = pd.read_csv('test_data.csv')

# print( list(df.loc[0,:]) )

dimensions = {'D1': ('K', 16), 'D2': ('C', 16), 'D3': ('OX', 2), 'D4': ('OY', 2),}

base_dims = [Dimension(idx, name, size) for idx, (name, size) in enumerate(dimensions.items())]
# dimensions_ = base_dims
# dimension_sizes = [dim.size for dim in base_dims]
# nb_dimensions = len(base_dims)

dimensions_new = {}
for dim_obj in base_dims:
    dimensions_new[dim_obj.name] = 1024 if dim_obj.name == 'D1' else 1
    
print(dimensions_new)