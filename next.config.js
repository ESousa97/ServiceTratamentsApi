// next.config.js
module.exports = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: process.env.NODE_ENV === 'production' ? 'https://your-vercel-deployment-url/api/:path*' : 'http://localhost:5328/:path*',
      },
    ]
  },
};
