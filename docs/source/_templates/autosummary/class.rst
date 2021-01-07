:github_url: {{ fullname | github_url }}

{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. add toctree option to make autodoc generate the pages

.. autoclass:: {{ objname }}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Attributes

    .. autosummary::
        :toctree: .
    {% for item in attributes %}
        {%- if item[0] != "_" %}
        ~{{ fullname }}.{{ item }}
        {%- endif -%}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block methods %}
    {% if methods %}
    .. rubric:: Methods

    .. autosummary::
        :toctree: .
    {% if objname != "CAResult" %}
        {% for item in methods %}
                {% if item[0] != "_" %}
                ~{{ fullname }}.{{ item }}
                {% endif %}
        {%- endfor %}
    {% else %}
        {% for item in methods %}
                {% if ((item[0] != "_") and (item not in inherited_members)) %}
                ~{{ fullname }}.{{ item }}
                {% endif %}
        {%- endfor %}
    {% endif %}
    {% endif %}
    {% endblock %}
