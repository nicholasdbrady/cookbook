import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';

import markdoc from "@astrojs/markdoc";

// https://astro.build/config
export default defineConfig({
  site: 'https://nicholasdbrady.github.io',
  //base: '/cookbook',
  trailingSlash: 'ignore',
  integrations: [mdx(), sitemap(), markdoc()]
});