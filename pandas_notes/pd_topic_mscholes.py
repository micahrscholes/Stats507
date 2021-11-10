#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 12:49:50 2021

@author: micahscholes
"""

# ## Pandas Topics
#
# Micah Scholes
# mscholes#umich.edu
#


# ## pandas.DataFrame.Insert()
#
# - The insert command is used to insert a new column to an existing dataframe.
# - While the merge command can also add columns to a dataframe, it is better for organizing data. The insert command works for data that is already organized where a column just needs to be added.
#
# ## Args
#
# The arguments for Insert() are:
# - loc: an integer representing the index location where the new column should be inserted
# - column: the name of the column. This should be a unique column name unless duplicates are desired.
# - value: a list, array, series, int, etc. of values to populate the column.
# - allow_duplicates: default is False. If set to true, it will allow you to insert a column with a duplicate name to an existing column.
#
# ## Example
#

import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8], 'b': [3, 4, 5, 6, 7, 8, 9,
                                                        10]})
df.insert(2,'c',["a", "b", "c", "d", "e", "f", "g", "h"])
df


# An error is raised if a duplicate column name is attempted to be inserted without setting allow_duplicates to true

df.insert(0,'a', [5, 6, 7, 8, 9,10, 11, 12])

df.insert(0,'a', [5, 6, 7, 8, 9,10, 11, 12], True)
df

# Additionally, the values have to be the same length as the other columns, otherwise we get an error.

df.insert(0,'d', [5, 6, 7, 8, 9,10, 11])

df.insert(0,'d', 1)
df

# However if only 1 value is entered, it will populate the entire column with that value.




