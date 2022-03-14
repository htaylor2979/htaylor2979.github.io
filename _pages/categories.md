---
layout: main
title: categories
---

<div id="archives">
{% for category in site.categories %}
  <div class="archive-group">
    {% capture category_name %}{{ category | first }}{% endcapture %}
    <div id="#{{ category_name | slugize }}"></div>
    <p></p>

    <h3 class="category-head">{{ category_name }}</h3>
    <a name="{{ category_name | slugize }}"></a>
    {% for post in site.categories[category_name] %}
    
      <article class="post">
        {% if post.img %}
          <a class="post-thumbnail" style="background-image: url({{"/assets/img/" | prepend: site.baseurl | append : post.img}})" href="{{post.url | prepend: site.baseurl}}"></a>
        {% else %}
        {% endif %}
        <div class="post-content">
          <h2 class="post-title"><a href="{{post.url | prepend: site.baseurl}}">{{post.title}}</a></h2>
          <p>{{ post.content | strip_html | truncatewords: 15 }}</p>
          <span class="post-date">{{post.date | date: '%Y, %b %d'}}&nbsp;&nbsp;&nbsp;—&nbsp;</span>
          <span class="post-words">{% capture words %}{{ post.content | number_of_words }}{% endcapture %}{% unless words contains "-" %}{{ words | plus: 250 | divided_by: 250 | append: " minute read" }}{% endunless %}</span>
        </div>
        {% if post.tags %}

          <div class="tags">
            <ul>
            {% for tag in post.tags %}
              <li>
                <a href="{{site.baseurl}}/tags#{{tag}}" class="tag">&#35; {{ tag }}</a>
              </li>
            {% endfor %}
            </ul>
          </div>
        {% else %}
        {% endif %}
      </article>
      
    {% endfor %}
  </div>
{% endfor %}
</div>