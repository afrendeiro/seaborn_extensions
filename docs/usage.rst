=====
Usage
=====

To use plotting functions of seaborn_extensions in a project::

    from seaborn_extensions import clustermap, swarmboxenplot


Simple examples::

    import pandas as pd
    from seaborn_extensions import swarmboxenplot

    data = pd.DataFrame(
        {"cont": np.random.random(20), "cat": np.random.choice(["a", "b"], 20)}
    )
    data.loc[data["cat"] == "b", "cont"] *= 5

    # A categorical variable vs a continuous variable
    fig, stats = swarmboxenplot(data=data, x='cat', y='cont')

    # A categorical variable vs a continuous variable stratified by another categorical variable
    fig, stats = swarmboxenplot(data=data, x='cat', y='cont', hue='h')

    # Plot of a categorical vs multiple continuous variables
    data['cont1'] = data['cont'] + np.random.random(20)
    data['cont2'] = data['cont'] + np.random.random(20)
    fig, stats = swarmboxenplot(data=data, x='cat', y=['cont1', 'cont2'], hue='h')


Check out the complete API at the :doc:`api` section.
