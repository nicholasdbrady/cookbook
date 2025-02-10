import rss from '@astrojs/rss';
import { getCollection } from 'astro:content';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';

export async function GET(context) {
	// Fetch all blog posts; filter out drafts (assuming you use a `draft` flag in frontmatter)
	const posts = await getCollection('blog', ({ data }) => !data.draft);

	// Get the absolute site URL from the context (configured in astro.config.mjs)
	const siteUrl = context.site;

	// Process each post to create an RSS feed item.
	const items = await Promise.all(posts.map(async (post) => {
		// Render the post (if necessary) to get the HTML. 
		// If your loader already produces HTML in post.body, you can skip this.
		const { Content } = await post.render();

		// If your frontmatter includes a heroImage (a relative path), convert it to an absolute URL
		const heroImageHTML = post.data.heroImage
			? `<p><img src="${new URL(post.data.heroImage, siteUrl).href}" alt="${post.data.title} Hero Image" /></p>`
			: '';

		return {
			title: post.data.title,
			// Construct the link using your collection's URL structure
			link: `/blog/${post.slug}/`,
			pubDate: post.data.pubDate,
			// Use a short summary as description; if you want to include the full content, set it in content
			description: post.data.description,
			// Prepend the hero image (if available) to the post body
			content: heroImageHTML + post.body,
			// Include categories if available (from your frontmatter, e.g., tags)
			categories: post.data.tags || [],
			// (Optional) Add an author field if you want
			author: post.data.author || undefined,
		};
	}));

	return rss({
		title: SITE_TITLE,                // Your feed title
		description: SITE_DESCRIPTION,    // A short description of your feed
		site: siteUrl,                    // The absolute base URL of your site
		items,                           // The array of RSS feed items you just created
		trailingSlash: false,
	});
}

