![](../docs/Images/logo_large.png)

# Blockly Earthquake Transformer
Blockly Earthquake Transformer (BET) was born from the need for seismologists and developers who are not AI specialists to easily, quickly, and independently build seismic phase pickers.

Blockly Earthquake Transformer (BET) is driven by the-state-of-art AI model - [EQTranformer(EqT)](https://github.com/smousavi05/EQTransformer) from [Mousavi et al. (2020)](https://www.nature.com/articles/s41467-020-17591-w.epdf?sharing_token=IiqAaF4NxwhUWGQLLLyTw9RgN0jAjWel9jnR3ZoTv0Nn-FaUKb3nu4lFkVXeZX_BCz5eMr5DkfCxQ3XASbeWwldzdU9oZF3d2MMG4cz6GWhVklzzzlL0QeMcf9kJJxA8wJAFfFCmtdlpQklDmGG7qRVjJxlCK-nusJjMFWE2oEk%3D).
# Get Started
### Open a terminal and enter notebook directory
```bash
cd notebook
jupyter notebook
```

### Option One: As a standalone application

To render the `blocklyeqt` example notebook as a standalone app, run
```bash
voila blocklyeqt.ipynb
```
To serve a directory of Jupyter notebooks, run `voila` with no argument.
![](../docs/Images/voila_start.gif)

### Option Two: Render the notebook within Jupyter
Voilà can also be used as a Jupyter server extension, both with the
[notebook](https://github.com/jupyter/notebook) server or with
[jupyter_server](https://github.com/jupyter/jupyter_server).

To install the Jupyter server extension, run

```bash
jupyter serverextension enable voila
jupyter server extension enable voila
```

When running the Jupyter server, the Voilà app is accessible from the base url uffixed with `voila`. Open `blocklyeqt.ipynb` via your local [Jupyter Notebook](https://jupyter.org/) or [Jupyter Lab]((https://jupyter.org/)). Click on `Voilà` icon [<img src="../docs/Images/jupytericon.png" width="30"/>](jupytericon.png)
