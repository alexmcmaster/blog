---
layout: default
title: Home
---

## Welcome to my blog

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <span>{{ post.date | date: "%B %d, %Y" }}</span>
      <p>{{ post.excerpt }}</p>
    </li>
  {% endfor %}
</ul>
