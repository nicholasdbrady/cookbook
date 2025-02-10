import rss, { pagesGlobToRssItems } from '@astrojs/rss';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';

export async function GET(context) {
  // Get the raw site URL from context (ensure it's a string) and remove any trailing slash.
  const rawSite = typeof context.site === 'string' ? context.site : context.site.href;
  const trimmedSite = rawSite.replace(/\/$/, '');
  
  // Our configured base path from astro.config.mjs
  const basePath = '/cookbook/';
  // Construct the full site URL (e.g., "https://nicholasdbrady.github.io/cookbook/")
  const fullSiteUrl = trimmedSite + basePath;

  // Use pagesGlobToRssItems() to automatically generate RSS items from your blog pages.
  // This assumes your published blog pages live in src/pages/blog/.
  let items = await pagesGlobToRssItems(import.meta.glob('./blog/*.{astro,md,mdx}'));

  // Filter out the dynamic route file (i.e. [...slug].astro) which doesn't have its own frontmatter.
  items = items.filter(item => !item.file.includes('[...slug]'));

  // Post-process each item so that its link and guid are built from the fullSiteUrl.
  items = items.map(item => {
    // Remove any leading slash from the item.link (if present)
    const relativePath = item.link.startsWith('/') ? item.link.substring(1) : item.link;
    // Build the final URL using fullSiteUrl as the base.
    const fixedLink = new URL(relativePath, fullSiteUrl).href;
    return {
      ...item,
      link: fixedLink,
      guid: fixedLink,
    };
  });

  return rss({
    title: SITE_TITLE,                   // Your feed title
    description: SITE_DESCRIPTION,       // Your feed description
    site: fullSiteUrl,                   // Ensures the channel <link> includes the base path
    items,                             // The list of RSS items with full HTML content
    trailingSlash: true,
    customData: `<language>en-us</language>`, // Optional extra XML data
  });
}
