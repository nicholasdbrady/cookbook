import rss, { pagesGlobToRssItems } from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';

export async function GET(context) {
  // Get the raw site URL and remove a trailing slash, if any.
  const rawSite =
    typeof context.site === 'string' ? context.site : context.site.href;
  const trimmedSite = rawSite.replace(/\/$/, '');

  // Append your base from astro.config.mjs.
  const basePath = '/cookbook/';
  const fullSiteUrl = trimmedSite + basePath; // e.g. "https://nicholasdbrady.github.io/cookbook/"

  // Fetch your blog posts from your content collection.
  const posts = await getCollection('blog', ({ data }) => !data.draft);

  // Process each post to create an RSS item.
  const items = await Promise.all(
    posts.map(async (post) => {
      const { Content } = await post.render();
      // Use fullSiteUrl to resolve heroImage.
      const heroImageHTML = post.data.heroImage
        ? `<p><img src="${new URL(post.data.heroImage, fullSiteUrl).href}" alt="${post.data.title} Hero Image" /></p>`
        : '';
      return {
        title: post.data.title,
        // Use fullSiteUrl to construct the item link.
        link: new URL(`blog/${post.slug}/`, fullSiteUrl).href,
        pubDate: post.data.pubDate,
        description: post.data.description,
        content: heroImageHTML + post.body,
        categories: post.data.tags || [],
        author: post.data.author || undefined,
      };
    })
  );

  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: fullSiteUrl, // This ensures the channel <link> is "https://nicholasdbrady.github.io/cookbook/"
    items,
    trailingSlash: true, // To match your astro.config.mjs trailingSlash: "always"
    customData: `<language>en-us</language>`,
  });
}
