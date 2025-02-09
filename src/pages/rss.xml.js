import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';

export async function GET(context) {
  // Retrieve all published blog posts from the "blog" collection
  const posts = await getCollection('blog', ({ data }) => !data.draft);

  // Map each post to an RSS feed item.
  // Here we include the full content from the post body.
  // (If you need to render Markdown to HTML, you may use post.render() with appropriate rendering logic.)
  const items = posts.map((post) => ({
    title: post.data.title,
    // Link to the blog post page; adjust if your routing differs.
    link: `/blog/${post.slug}/`,
    // Publication date from frontmatter
    pubDate: post.data.pubDate,
    // A short summary/description for the feed item
    description: post.data.description,
    // Full post content – note that post.body is the raw Markdown.
    // For proper HTML, ensure your collection loader converts it, or use post.render() to generate HTML.
    content: post.body,
    // Optional: add categories if available
    categories: post.data.tags || [],
    // Optionally, include the author field if present
    author: post.data.author || undefined,
  }));

  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site,
    items,
  });
}
