import rss, { pagesGlobToRssItems } from '@astrojs/rss';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';

export async function GET(context) {
  // Ensure context.site is a string and remove any trailing slash.
  const rawSite = typeof context.site === 'string' ? context.site : context.site.href;
  const trimmedSite = rawSite.replace(/\/$/, '');
  const basePath = '/cookbook/';
  // Construct the full site URL: "https://nicholasdbrady.github.io/cookbook/"
  const fullSiteUrl = trimmedSite + basePath;

  // Use a glob pattern array to include only blog pages and exclude the dynamic route file.
  let items = await pagesGlobToRssItems(
    import.meta.glob([
      './blog/*.{astro,md,mdx}',
      '!./blog/[...slug].astro'
    ])
  );

  // Post-process each item so that its link and guid are built correctly.
  items = items.map(item => {
    // Remove a leading slash from the item.link if present.
    const relativePath = item.link.startsWith('/') ? item.link.substring(1) : item.link;
    // Construct the fixed link using the fullSiteUrl as the base.
    const fixedLink = new URL(relativePath, fullSiteUrl).href;
    return {
      ...item,
      link: fixedLink,
      guid: fixedLink,
    };
  });

  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: fullSiteUrl, // Ensures the channel <link> includes the "/cookbook/" base.
    items,
    trailingSlash: true, // Matches your astro.config.mjs trailingSlash: "always"
    customData: `<language>en-us</language>`,
  });
}
