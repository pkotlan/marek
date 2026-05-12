import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "MAReK",
  description: "Simple GUI for image annotation",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      // { text: 'Home', link: '/' },
      { text: 'Usage', link: '/usage' }
    ],

    sidebar: [
      {
        text: 'Getting started',
        items: [
          { text: 'Installation', link: '/installation' },
          { text: 'Usage', link: '/usage' }
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/pkotlan/marek' }
    ]
  }
})
