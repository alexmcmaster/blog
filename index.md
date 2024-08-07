---
layout: default
title: Home
---

## Welcome to my blog

<html>
  <head>

  <!-- Google tag (gtag.js) -->
  <script async src="https://www.googletagmanager.com/gtag/js?id=G-NG81F08RSR"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
  
    gtag('config', 'G-NG81F08RSR');
  </script>

  </head>
  <body>
    <ul>
      {% for post in site.posts %}
        <li>
          <a href="{{ site.baseurl}}{{ post.url }}">{{ post.title }}</a>
          <span>{{ post.date | date: "%B %d, %Y" }}</span>
          <p>{{ post.excerpt }}</p>
        </li>
      {% endfor %}
    </ul>
  </body>
</html>
